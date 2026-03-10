"""
api/main.py
FastAPI gateway — multi-tenant RAG endpoints with JWT-based tenant isolation.
"""
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import jwt

from ingestion.pipeline import ingest_document, delete_document
from inference.rag_pipeline import query

app = FastAPI(title="Multi-Tenant RAG API", version="1.0.0")

SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key")
ALGORITHM = "HS256"


# ── Auth ──────────────────────────────────────────────────────────────────────

class TenantContext(BaseModel):
    tenant_id: str
    tier: str = "free"   # free | pro | enterprise


def get_tenant(authorization: str = Header(...)) -> TenantContext:
    """
    Extract tenant_id and tier from JWT token.
    This is the security boundary — tenant_id comes from the verified token,
    NOT from the request body. Users cannot spoof another tenant's ID.
    """
    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TenantContext(
            tenant_id=payload["tenant_id"],
            tier=payload.get("tier", "free"),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Request/Response Models ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    use_hybrid: bool = True        # BM25 + kNN vs pure kNN

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    tenant_id: str
    cached: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest", summary="Upload and index a document")
async def ingest(
    file: UploadFile = File(...),
    tenant: TenantContext = Depends(get_tenant),
):
    """
    Upload a document (PDF, TXT, DOCX, CSV) to your personal knowledge base.
    The file is chunked, embedded, and stored in OpenSearch under your tenant_id.
    """
    allowed_types = {".pdf", ".txt", ".docx", ".csv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_types:
        raise HTTPException(400, f"File type {ext} not supported. Use: {allowed_types}")

    # Save upload to temp file, then ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_document(
            file_path=tmp_path,
            tenant_id=tenant.tenant_id,
            tier=tenant.tier,
        )
    finally:
        os.unlink(tmp_path)

    return result


@app.post("/query", response_model=QueryResponse, summary="Query your knowledge base")
async def query_kb(
    request: QueryRequest,
    tenant: TenantContext = Depends(get_tenant),
):
    """
    Ask a question against YOUR knowledge base only.
    Results are strictly filtered to your tenant_id — zero cross-tenant leakage.
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    result = query(
        question=request.question,
        tenant_id=tenant.tenant_id,
        tier=tenant.tier,
        use_hybrid=request.use_hybrid,
    )
    return result


@app.delete("/documents/{doc_id}", summary="Delete a document from your knowledge base")
async def delete_doc(
    doc_id: str,
    tenant: TenantContext = Depends(get_tenant),
):
    """
    Delete all chunks of a document. Only deletes within your tenant_id.
    """
    result = delete_document(doc_id, tenant.tenant_id, tenant.tier)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Dev Runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
