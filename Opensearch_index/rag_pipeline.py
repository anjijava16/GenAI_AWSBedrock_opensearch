"""
inference/rag_pipeline.py
Multi-tenant RAG inference: embed query → filtered kNN → LLM answer.
"""
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from core.opensearch_client import get_vector_store, get_os_client, get_index_name
from typing import List
import hashlib, json

# Optional: Redis for caching
try:
    import redis
    cache = redis.Redis(host="localhost", port=6379, decode_responses=True)
    CACHE_TTL = 3600  # 1 hour
    CACHE_ENABLED = True
except Exception:
    CACHE_ENABLED = False


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""",
)


# ── Tenant-Aware Retriever ────────────────────────────────────────────────────

class TenantFilteredRetriever(BaseRetriever):
    """
    LangChain retriever that ALWAYS applies a tenant_id pre-filter.
    This prevents any cross-tenant data leakage.
    """
    tenant_id: str
    tier: str = "free"
    k: int = 5
    score_threshold: float = 0.3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vs = get_vector_store(self.tenant_id, self.tier)

        # LangChain's similarity_search with metadata pre-filter
        # The filter is enforced at the OpenSearch query level, not post-retrieval
        results = vs.similarity_search_with_score(
            query,
            k=self.k,
            pre_filter={
                "bool": {
                    "filter": [
                        {"term": {"metadata.tenant_id": self.tenant_id}}
                    ]
                }
            },
        )

        # Optionally filter by relevance score
        return [
            doc for doc, score in results
            if score >= self.score_threshold
        ]


# ── Hybrid Search (BM25 + kNN) ────────────────────────────────────────────────

def hybrid_search(query: str, tenant_id: str, tier: str = "free", k: int = 5):
    """
    Combines BM25 keyword search + kNN vector search for better recall.
    Best of both worlds: exact keyword matches + semantic similarity.
    """
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = embeddings.embed_query(query)

    index = get_index_name(tenant_id, tier)
    client = get_os_client()

    hybrid_query = {
        "size": k,
        "_source": {"excludes": ["vector_field"]},
        "query": {
            "bool": {
                "filter": [
                    {"term": {"metadata.tenant_id": tenant_id}}  # ← always filter
                ],
                "should": [
                    # BM25 keyword match
                    {"match": {"text": {"query": query, "boost": 0.3}}},
                    # kNN vector similarity
                    {"knn": {"vector_field": {"vector": query_vector, "k": k, "boost": 0.7}}},
                ],
                "minimum_should_match": 1,
            }
        },
    }

    response = client.search(index=index, body=hybrid_query)
    hits = response["hits"]["hits"]

    return [
        Document(
            page_content=hit["_source"]["text"],
            metadata=hit["_source"].get("metadata", {}),
        )
        for hit in hits
    ]


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def build_rag_chain(tenant_id: str, tier: str = "free") -> RetrievalQA:
    """
    Builds a full LangChain RAG chain for a specific tenant.
    The retriever automatically isolates to this tenant's data.
    """
    retriever = TenantFilteredRetriever(tenant_id=tenant_id, tier=tier)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )


def query(
    question: str,
    tenant_id: str,
    tier: str = "free",
    use_hybrid: bool = True,
) -> dict:
    """
    Main entry point for RAG queries.

    1. Check cache
    2. Retrieve relevant chunks (filtered to tenant)
    3. Generate answer via LLM
    4. Cache result
    """
    # Cache key = hash of (tenant_id, question)
    cache_key = f"rag:{hashlib.md5(f'{tenant_id}:{question}'.encode()).hexdigest()}"

    if CACHE_ENABLED:
        cached = cache.get(cache_key)
        if cached:
            return {**json.loads(cached), "cached": True}

    if use_hybrid:
        # Hybrid search manually, then pass to LLM
        docs = hybrid_search(question, tenant_id, tier)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = llm.invoke(prompt).content
        sources = [d.metadata.get("source", "unknown") for d in docs]
    else:
        chain = build_rag_chain(tenant_id, tier)
        result = chain.invoke({"query": question})
        answer = result["result"]
        sources = list({d.metadata.get("source") for d in result["source_documents"]})

    response = {
        "answer": answer,
        "sources": sources,
        "tenant_id": tenant_id,
        "cached": False,
    }

    if CACHE_ENABLED:
        cache.setex(cache_key, CACHE_TTL, json.dumps(response))

    return response
