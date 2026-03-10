"""
ingestion/pipeline.py
Multi-tenant document ingestion: load → chunk → embed → store in OpenSearch.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from langchain_core.documents import Document

from core.opensearch_client import get_vector_store, get_os_client


# ── Text Splitter ─────────────────────────────────────────────────────────────
# chunk_size: tokens per chunk (≈ 400 words)
# chunk_overlap: overlap to preserve context at boundaries
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_document(file_path: str) -> List[Document]:
    """Detect file type and load with appropriate LangChain loader."""
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".csv":  CSVLoader,
    }
    loader_cls = loaders.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(file_path).load()


def ingest_document(
    file_path: str,
    tenant_id: str,
    tier: str = "free",
    extra_metadata: dict = None,
) -> dict:
    """
    Full ingestion pipeline for one document.

    Args:
        file_path:  Local path to the uploaded file.
        tenant_id:  Identifies the tenant/user (injected from JWT).
        tier:       'free' | 'pro' | 'enterprise' — controls index routing.
        extra_metadata: Any additional metadata to store with chunks.

    Returns:
        Summary dict with chunk count and index name.
    """
    # 1. Load
    raw_docs = load_document(file_path)

    # 2. Split into chunks
    chunks = SPLITTER.split_documents(raw_docs)

    # 3. Inject tenant metadata into EVERY chunk
    #    This is the critical step that enables multi-tenant filtering later.
    doc_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "tenant_id": tenant_id,          # ← mandatory for isolation
            "source": Path(file_path).name,
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_{i}",
            "created_at": datetime.utcnow().isoformat(),
            **(extra_metadata or {}),
        })

    # 4. Embed + store via LangChain
    vs = get_vector_store(tenant_id, tier)
    vs.add_documents(chunks)

    return {
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "chunks_ingested": len(chunks),
        "index": vs.index_name,
        "source": Path(file_path).name,
    }


def delete_document(doc_id: str, tenant_id: str, tier: str = "free"):
    """
    Delete all chunks belonging to a specific document for a tenant.
    Uses OpenSearch delete-by-query — safe because tenant_id is always a filter.
    """
    from core.opensearch_client import get_index_name
    client = get_os_client()
    index = get_index_name(tenant_id, tier)

    response = client.delete_by_query(
        index=index,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"metadata.tenant_id": tenant_id}},
                        {"term": {"metadata.doc_id": doc_id}},
                    ]
                }
            }
        },
    )
    return {"deleted": response["deleted"]}
