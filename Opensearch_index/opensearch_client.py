"""
core/opensearch_client.py
Central OpenSearch connection and index management for multi-tenant RAG.
"""
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings
import os

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
OS_USER = os.getenv("OS_USER", "admin")
OS_PASS = os.getenv("OS_PASS", "admin")

# Shared index for free/pro tenants
SHARED_INDEX = "kb_shared_pool"
# Enterprise tenants get: kb_enterprise_{tenant_id}

VECTOR_DIMENSION = 1536  # OpenAI text-embedding-3-small

KNN_INDEX_SETTINGS = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512,
            "number_of_shards": 2,
            "number_of_replicas": 1,
        }
    },
    "mappings": {
        "properties": {
            "vector_field": {
                "type": "knn_vector",
                "dimension": VECTOR_DIMENSION,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "faiss",
                    "parameters": {"ef_construction": 512, "m": 16},
                },
            },
            "text": {"type": "text", "analyzer": "standard"},
            "metadata": {
                "properties": {
                    "tenant_id": {"type": "keyword"},   # ← key for isolation
                    "source": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "created_at": {"type": "date"},
                }
            },
        }
    },
}


def get_os_client() -> OpenSearch:
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OS_USER, OS_PASS),
        use_ssl=True,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )


def get_index_name(tenant_id: str, tier: str = "free") -> str:
    """
    Routing logic:
      - enterprise  → dedicated index:  kb_enterprise_{tenant_id}
      - free / pro  → shared index:     kb_shared_pool
    """
    if tier == "enterprise":
        return f"kb_enterprise_{tenant_id}"
    return SHARED_INDEX


def ensure_index_exists(index_name: str, client: OpenSearch = None):
    """Create the index with kNN mapping if it doesn't exist."""
    client = client or get_os_client()
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=KNN_INDEX_SETTINGS)
        print(f"✅ Created index: {index_name}")


def get_vector_store(tenant_id: str, tier: str = "free") -> OpenSearchVectorSearch:
    """
    Returns an OpenSearchVectorSearch bound to the correct index for this tenant.
    LangChain handles embedding + storage automatically.
    """
    index_name = get_index_name(tenant_id, tier)
    ensure_index_exists(index_name)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL,
        http_auth=(OS_USER, OS_PASS),
        use_ssl=True,
        verify_certs=False,
    )
