# Input Opensearch Schema 
```**Notes:**

"""OpenSearch index configuration for hybrid search (BM25 + Vector).

This configuration supports both keyword search (BM25) and vector similarity search
using HNSW algorithm for approximate nearest neighbor search.
"""

ARXIV_PAPERS_CHUNKS_INDEX = "arxiv-papers-chunks"

# Index mapping for chunked papers with vector embeddings
ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "index.knn.space_type": "cosinesimil",
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                "text_analyzer": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "stop", "snowball"]},
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "chunk_word_count": {"type": "integer"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # Jina v3 embeddings dimension
                "method": {
                    "name": "hnsw",  # Hierarchical Navigable Small World
                    "space_type": "cosinesimil",  # Cosine similarity
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,  # Higher value = better recall, slower indexing
                        "m": 16,  # Number of bi-directional links
                    },
                },
            },
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text", "analyzer": "text_analyzer"},
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "section_title": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}

HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",  # Reciprocal Rank Fusion
                    "rank_constant": 60,  # Default k=60 for RRF formula: 1/(k+rank)
                }
            }
        }
    ],
}


```

## **1. Overview**

This OpenSearch setup is designed for **arXiv paper chunks**. Each paper is split into chunks (sections or paragraphs), and each chunk has:

1. **Textual fields** (`title`, `chunk_text`, `abstract`, `authors`) → for **keyword search** using BM25.
2. **Vector embeddings** (`embedding`) → for **semantic search** using **Approximate Nearest Neighbor (ANN)** algorithms, specifically **HNSW** via **nmslib**.
3. **Metadata fields** (`arxiv_id`, `categories`, `published_date`, etc.) → for filtering or faceting.

This combination allows **hybrid search**: you can retrieve documents both by relevance to keywords **and** semantic similarity.

---

## **2. Index Settings**

```python
"settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index.knn": True,
    "index.knn.space_type": "cosinesimil",
    ...
}
```

### Explanation:

* `number_of_shards: 1`

  * Simplifies deployment for smaller datasets.
  * Sharding can help scale horizontally, but one shard is sufficient for moderate-scale research corpora.

* `number_of_replicas: 0`

  * No replication, likely for local or testing environments.
  * Production usually uses ≥1 replica for redundancy.

* `index.knn: True`

  * Enables **K-Nearest Neighbor search** for vector fields.
  * Without this, you cannot query embeddings efficiently.

* `index.knn.space_type: "cosinesimil"`

  * Defines the **distance metric** for ANN search.
  * Cosine similarity is common for high-dimensional embeddings from models like **Jina v3**.

---

## **3. Text Analysis Configuration**

```python
"analysis": {
    "analyzer": {
        "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
        "text_analyzer": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "stop", "snowball"]
        },
    }
}
```

* `standard_analyzer`

  * Uses built-in tokenization + English stopword removal.
  * Ideal for simple fields like `authors` where stemming is unnecessary.

* `text_analyzer`

  * Custom analyzer for heavy text fields (`chunk_text`, `title`, `abstract`).
  * Components:

    1. **Tokenizer:** Standard (splits by whitespace/punctuation)
    2. **Filters:**

       * `lowercase` → normalize to lowercase
       * `stop` → removes common stopwords (“the”, “and”, etc.)
       * `snowball` → applies **Porter stemmer variant**, reducing words to their root form (“running” → “run”)

✅ This improves **BM25 relevance** by unifying word forms.

---

## **4. Mappings**

Mappings define **how each field is stored, indexed, and searched**.

### **4.1 Identifiers & metadata**

```python
"chunk_id": {"type": "keyword"},
"arxiv_id": {"type": "keyword"},
"paper_id": {"type": "keyword"},
"chunk_index": {"type": "integer"},
```

* `keyword` → exact match (not tokenized).
* Used for:

  * Filtering (`arxiv_id:1234`)
  * Sorting (`chunk_index`)

`chunk_index` tracks the order of chunks within a paper.

---

### **4.2 Text fields (BM25 search)**

```python
"chunk_text": {
    "type": "text",
    "analyzer": "text_analyzer",
    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
},
```

* `type: text` → full-text search enabled (BM25).
* `analyzer: text_analyzer` → stemmed + lowercased for better relevance.
* `fields.keyword` → allows exact matches or aggregations (e.g., group chunks by identical content).

**Other text fields:**

* `title`, `abstract` → similar to `chunk_text`, suitable for BM25.
* `authors` → `standard_analyzer`, mainly for exact name matching + simple search.

---

### **4.3 Vector embedding field (semantic search)**

```python
"embedding": {
    "type": "knn_vector",
    "dimension": 1024,
    "method": {
        "name": "hnsw",
        "space_type": "cosinesimil",
        "engine": "nmslib",
        "parameters": {
            "ef_construction": 512,
            "m": 16,
        },
    },
}
```

#### Deep Dive:

* `type: knn_vector` → declares the field stores high-dimensional embeddings for ANN.

* `dimension: 1024` → matches **Jina v3 embedding size**.

* **HNSW (Hierarchical Navigable Small World)**:

  * Graph-based ANN algorithm for fast nearest neighbor search.
  * Efficient even for millions of vectors.
  * Key parameters:

    * `m: 16` → number of connections per node. Higher = better accuracy, more memory.
    * `ef_construction: 512` → higher = better recall, slower index building.

* `space_type: cosinesimil` → measures cosine similarity between embeddings.

* `engine: nmslib` → OpenSearch supports multiple ANN backends; nmslib is fast and memory-efficient.

✅ This is the **core of semantic search**, allowing retrieval of semantically similar chunks even without exact keyword matches.

---

### **4.4 Dates, categories, and other metadata**

```python
"categories": {"type": "keyword"},
"published_date": {"type": "date"},
"created_at": {"type": "date"},
"updated_at": {"type": "date"},
```

* Used for filtering, faceting, or sorting.
* `keyword` → exact category matches (`physics`, `cs.AI`).
* `date` → supports range queries (`2020-01-01` → `2021-01-01`).

---

## **5. Hybrid Search Pipeline (RRF)**

```python
HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",
                    "rank_constant": 60,
                }
            }
        }
    ],
}
```

### Explanation:

* **Hybrid Search Problem**: You have two different scores for the same document:

  1. BM25 keyword relevance
  2. Vector similarity (embedding cosine similarity)

* **Reciprocal Rank Fusion (RRF)**:

  * Combines multiple ranked lists into one unified ranking.
  * Formula:
    [
    score = \sum_i \frac{1}{k + rank_i}
    ]
    where `rank_i` = rank of a document in source `i`, `k` = `rank_constant` (here 60).

* `rank_constant: 60` → prevents top-ranked results from dominating too heavily.

✅ Using this, documents that score moderately well in both BM25 and vector search can surface to the top, improving **hybrid relevance**.

---

## **6. Key Design Choices and Rationale**

| Feature         | Why It Matters                                        |
| --------------- | ----------------------------------------------------- |
| Chunked papers  | Enables fine-grained semantic search.                 |
| `knn_vector`    | Allows fast semantic similarity search on embeddings. |
| `text_analyzer` | Stemmed + stopword filtered BM25 search.              |
| HNSW parameters | Balanced accuracy vs. index speed & memory.           |
| Hybrid RRF      | Combines BM25 + vector scores robustly.               |
| Metadata        | Enables filtering, faceting, and sorting.             |

---

## ✅ **Summary**

This schema supports a **true hybrid search**:

1. **Keyword (BM25)**: `chunk_text`, `title`, `abstract`, etc.
2. **Semantic (Vector)**: `embedding` via HNSW ANN.
3. **Hybrid Fusion**: RRF merges BM25 + vector ranks.
4. **Metadata & filters**: For precise faceted search (dates, categories, authors).

It’s **scalable**, **efficient**, and **flexible**, tailored for academic research corpora like arXiv.

---

If you want, I can **draw a diagram showing the hybrid search flow**, from query → BM25 → vector → RRF → final ranked results, which makes this super clear.

Do you want me to do that?



## **1. `type`**

```json
"type": "knn_vector"
```

* `knn_vector` tells OpenSearch this field stores **high-dimensional vectors** (embeddings).
* These are not standard numeric fields — OpenSearch will build a specialized index to allow **approximate nearest neighbor (ANN)** search.
* Only `knn_vector` fields can be queried with `knn` queries.

**Why important:**
Without `knn_vector`, the index cannot perform **vector similarity searches**.

---

## **2. `dimension`**

```json
"dimension": 1024
```

* Specifies the **length of the embedding vector**.
* Must exactly match the dimensionality of the vectors you insert.

  * Example: if you generate embeddings with a model like **Jina v3** or **OpenAI embeddings**, you must set this correctly.
* OpenSearch uses this to allocate memory and compute distances.

**Tip:** If you insert vectors of the wrong size, indexing will fail.

---

## **3. `method`**

This is the core ANN configuration. It has **four subfields**:

---

### **3.1 `name`**

```json
"name": "hnsw"
```

* Specifies the ANN algorithm. Common options in OpenSearch:

  * `"hnsw"` → **Hierarchical Navigable Small World graph**
  * `"brute_force"` → exact vector search (slow, not recommended for large datasets)
  * `"ivf"` (in some implementations) → Inverted File index (common in FAISS)

**HNSW** is popular because:

* Very fast approximate search even for millions of vectors.
* Supports dynamic insertion.
* Provides tunable trade-off between speed and recall via `m` and `ef_construction`.

---

### **3.2 `space_type`**

```json
"space_type": "cosinesimil"
```

* Defines **distance/similarity metric**. Common types:

  | Space Type    | Meaning            | Use Case                                                     |
  | ------------- | ------------------ | ------------------------------------------------------------ |
  | `l2`          | Euclidean distance | For dense embeddings where magnitude matters                 |
  | `cosinesimil` | Cosine similarity  | Most NLP embeddings, normalized vectors                      |
  | `dot`         | Dot product        | Sometimes used for unnormalized embeddings in recommendation |

* **Cosine similarity** is standard for semantic embeddings because it measures **angle between vectors**, ignoring magnitude.

**Key:** The choice must match how your embeddings were generated.

---

### **3.3 `engine`**

```json
"engine": "nmslib"
```

* ANN **backend engine** used by OpenSearch. Options include:

  * `nmslib` → popular, memory-efficient, supports HNSW
  * `faiss` → another ANN engine, GPU optimized (in some distributions)
  * `lucene` → limited ANN support in classic Lucene indices

* `nmslib` is lightweight, fast, widely used for HNSW graphs.

**Why it matters:**
Engine choice affects **search speed**, **memory usage**, and **scalability**.

---

### **3.4 `parameters`**

```json
"parameters": {
    "ef_construction": 512,
    "m": 16
}
```

These are **HNSW-specific tuning parameters**:

| Parameter         | Meaning                                                      | Effect                                                                          |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `m`               | Number of bi-directional links per node in HNSW graph        | Higher `m` → better recall, more memory, slower indexing                        |
| `ef_construction` | Size of the dynamic candidate list during index construction | Higher → better graph connectivity → better search accuracy but slower indexing |

**Optional runtime parameter (not here)**:

* `ef_search` → candidate list size during query. Higher → better recall, slower search.

**Rules of thumb:**

* `m` ≈ 16–64 for medium datasets
* `ef_construction` ≈ 200–512
* Adjust for **tradeoff between indexing time, memory, and search quality**.

---

## **Summary Table**

| Field                               | Purpose                    | Typical Values / Notes           |
| ----------------------------------- | -------------------------- | -------------------------------- |
| `type`                              | Declares vector storage    | `"knn_vector"`                   |
| `dimension`                         | Embedding size             | e.g., 1024                       |
| `method.name`                       | ANN algorithm              | `"hnsw"`, `"brute_force"`        |
| `method.space_type`                 | Similarity metric          | `"cosinesimil"`, `"l2"`, `"dot"` |
| `method.engine`                     | Backend library            | `"nmslib"`, `"faiss"`            |
| `method.parameters.m`               | HNSW node connections      | 16–64                            |
| `method.parameters.ef_construction` | Graph build candidate list | 200–512                          |

✅ Correct tuning ensures **fast, accurate hybrid search** combining BM25 + vector similarity.

---

If you want, I can make a **visual diagram of HNSW indexing + search flow** showing what `m` and `ef_construction` actually do inside the graph — it makes this **much easier to intuitively understand**.

Do you want me to do that?

# Opensearch Client code

```python
"""Unified OpenSearch client supporting both simple BM25 and hybrid search."""

import logging
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from src.config import Settings

from .index_config_hybrid import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client supporting BM25 and hybrid search with native RRF."""

    def __init__(self, host: str, settings: Settings):
        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"

        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )

        logger.info(f"OpenSearch client initialized with host: {host}")

    def health_check(self) -> bool:
        """Check if OpenSearch cluster is healthy."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the hybrid index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"index_name": self.index_name, "exists": False, "document_count": 0}

            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]

            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        results = {}
        results["hybrid_index"] = self._create_hybrid_index(force)
        results["rrf_pipeline"] = self._create_rrf_pipeline(force)
        return results

    def _create_hybrid_index(self, force: bool = False) -> bool:
        """Create hybrid index for all search types (BM25, vector, hybrid).

        :param force: If True, recreate index even if it exists
        :returns: True if created, False if already exists
        """
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing hybrid index: {self.index_name}")

            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=ARXIV_PAPERS_CHUNKS_MAPPING)
                logger.info(f"Created hybrid index: {self.index_name}")
                return True

            logger.info(f"Hybrid index already exists: {self.index_name}")
            return False

        except Exception as e:
            logger.error(f"Error creating hybrid index: {e}")
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """Create RRF search pipeline for native hybrid search.

        :param force: If True, recreate pipeline even if it exists
        :returns: True if created, False if already exists
        """
        try:
            pipeline_id = HYBRID_RRF_PIPELINE["id"]

            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info(f"Deleted existing RRF pipeline: {pipeline_id}")
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info(f"RRF pipeline already exists: {pipeline_id}")
                return False
            except Exception:
                pass
            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }

            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body)

            logger.info(f"Created RRF search pipeline: {pipeline_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating RRF pipeline: {e}")
            raise

    def search_papers(
        self, query: str, size: int = 10, from_: int = 0, categories: Optional[List[str]] = None, latest: bool = True
    ) -> Dict[str, Any]:
        """BM25 search for papers."""
        return self._search_bm25_only(query=query, size=size, from_=from_, categories=categories, latest=latest)

    def search_chunks_vector(
        self, query_embedding: List[float], size: int = 10, categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Pure vector search on chunks.

        :param query_embedding: Query embedding vector
        :param size: Number of results
        :param categories: Optional category filter
        :returns: Search results
        """
        try:
            # Build filter
            filter_clause = []
            if categories:
                filter_clause.append({"terms": {"categories": categories}})

            search_body = {
                "size": size,
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": size}}},
                "_source": {"excludes": ["embedding"]},
            }

            if filter_clause:
                search_body["query"] = {"bool": {"must": [search_body["query"]], "filter": filter_clause}}

            response = self.client.search(index=self.index_name, body=search_body)

            results = {"total": response["hits"]["total"]["value"], "hits": []}

            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["score"] = hit["_score"]
                chunk["chunk_id"] = hit["_id"]
                results["hits"].append(chunk)

            return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"total": 0, "hits": []}

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Unified search method supporting BM25, vector, and hybrid modes.

        :param query: Text query for search
        :param query_embedding: Optional embedding for vector/hybrid search
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param categories: Optional category filter
        :param latest: Sort by date instead of relevance
        :param use_hybrid: If True and embedding provided, use hybrid search
        :param min_score: Minimum score threshold
        :returns: Search results
        """
        try:
            # If no embedding provided or hybrid disabled, use BM25 only
            if not query_embedding or not use_hybrid:
                return self._search_bm25_only(query=query, size=size, from_=from_, categories=categories, latest=latest)

            # Use native OpenSearch hybrid search with RRF pipeline
            return self._search_hybrid_native(
                query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
            )

        except Exception as e:
            logger.error(f"Unified search error: {e}")
            return {"total": 0, "hits": []}

    def _search_bm25_only(
        self, query: str, size: int, from_: int, categories: Optional[List[str]], latest: bool
    ) -> Dict[str, Any]:
        """Pure BM25 search implementation."""
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest_papers=latest,
            search_chunks=True,  # Enable chunk search mode
        )
        search_body = builder.build()

        response = self.client.search(index=self.index_name, body=search_body)

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        logger.info(f"BM25 search for '{query[:50]}...' returned {results['total']} results")
        return results

    def _search_hybrid_native(
        self, query: str, query_embedding: List[float], size: int, categories: Optional[List[str]], min_score: float
    ) -> Dict[str, Any]:
        """Native OpenSearch hybrid search with RRF pipeline."""
        builder = QueryBuilder(
            query=query, size=size * 2, from_=0, categories=categories, latest_papers=False, search_chunks=True
        )
        bm25_search_body = builder.build()

        bm25_query = bm25_search_body["query"]

        hybrid_query = {"hybrid": {"queries": [bm25_query, {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}]}}

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        # Execute search with RRF pipeline
        response = self.client.search(
            index=self.index_name, body=search_body, params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]}
        )

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue

            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        results["total"] = len(results["hits"])
        logger.info(f"Native hybrid search for '{query[:50]}...' returned {results['total']} results")
        return results

    def search_chunks_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        size: int = 10,
        categories: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Hybrid search combining BM25 and vector similarity using native RRF."""
        return self._search_hybrid_native(
            query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
        )

    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """Index a single chunk with its embedding.

        :param chunk_data: Chunk data dictionary
        :param embedding: Embedding vector
        :returns: True if successful
        """
        try:
            chunk_data["embedding"] = embedding

            response = self.client.index(index=self.index_name, body=chunk_data, refresh=True)

            return response["result"] in ["created", "updated"]

        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple chunks with embeddings.

        :param chunks: List of dicts with 'chunk_data' and 'embedding'
        :returns: Statistics
        """
        from opensearchpy import helpers

        try:
            actions = []
            for chunk in chunks:
                chunk_data = chunk["chunk_data"].copy()
                chunk_data["embedding"] = chunk["embedding"]

                action = {"_index": self.index_name, "_source": chunk_data}
                actions.append(action)

            success, failed = helpers.bulk(self.client, actions, refresh=True)

            logger.info(f"Bulk indexed {success} chunks, {len(failed)} failed")
            return {"success": success, "failed": len(failed)}

        except Exception as e:
            logger.error(f"Bulk chunk indexing error: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str) -> bool:
        """Delete all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: True if deletion was successful
        """
        try:
            response = self.client.delete_by_query(
                index=self.index_name, body={"query": {"term": {"arxiv_id": arxiv_id}}}, refresh=True
            )

            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for paper {arxiv_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def get_chunks_by_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: List of chunks sorted by chunk_index
        """
        try:
            search_body = {
                "query": {"term": {"arxiv_id": arxiv_id}},
                "size": 1000,
                "sort": [{"chunk_index": "asc"}],
                "_source": {"excludes": ["embedding"]},
            }

            response = self.client.search(index=self.index_name, body=search_body)

            chunks = []
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["chunk_id"] = hit["_id"]
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
```
