import { useState } from "react";

const colors = {
  bg: "#0a0e1a",
  surface: "#111827",
  border: "#1e293b",
  accent: "#3b82f6",
  green: "#10b981",
  orange: "#f59e0b",
  purple: "#8b5cf6",
  red: "#ef4444",
  text: "#e2e8f0",
  muted: "#64748b",
};

const Box = ({ color = colors.accent, title, items = [], icon, style = {} }) => (
  <div style={{
    border: `1.5px solid ${color}33`,
    borderRadius: 10,
    padding: "12px 16px",
    background: `${color}0d`,
    minWidth: 160,
    ...style
  }}>
    <div style={{ fontSize: 11, fontWeight: 700, color, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
      {icon && <span>{icon}</span>}
      {title}
    </div>
    {items.map((item, i) => (
      <div key={i} style={{ fontSize: 12, color: colors.text, padding: "3px 0", borderBottom: i < items.length - 1 ? `1px solid ${colors.border}` : "none" }}>
        {item}
      </div>
    ))}
  </div>
);

const Arrow = ({ label, vertical = false }) => (
  <div style={{
    display: "flex",
    flexDirection: vertical ? "column" : "row",
    alignItems: "center",
    gap: 4,
    color: colors.muted,
    fontSize: 10,
    fontFamily: "monospace",
  }}>
    {!vertical && <div style={{ width: 30, height: 1, background: colors.muted }} />}
    {vertical && <div style={{ width: 1, height: 20, background: colors.muted, alignSelf: "center" }} />}
    <span style={{ whiteSpace: "nowrap" }}>{label}</span>
    {!vertical && <div style={{ width: 30, height: 1, background: colors.muted }} />}
    {vertical && <div style={{ width: 1, height: 20, background: colors.muted, alignSelf: "center" }} />}
  </div>
);

const TenantBadge = ({ name, color }) => (
  <div style={{
    display: "inline-flex", alignItems: "center", gap: 6,
    background: `${color}22`, border: `1px solid ${color}55`,
    borderRadius: 20, padding: "3px 10px", fontSize: 11, color,
  }}>
    <div style={{ width: 6, height: 6, borderRadius: "50%", background: color }} />
    {name}
  </div>
);

const Tab = ({ label, active, onClick }) => (
  <button onClick={onClick} style={{
    background: active ? colors.accent : "transparent",
    border: `1px solid ${active ? colors.accent : colors.border}`,
    color: active ? "#fff" : colors.muted,
    borderRadius: 6, padding: "6px 14px", fontSize: 12,
    cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
  }}>{label}</button>
);

const CodeBlock = ({ code }) => (
  <pre style={{
    background: "#050810", border: `1px solid ${colors.border}`,
    borderRadius: 8, padding: 16, fontSize: 11.5,
    color: "#a5f3fc", overflowX: "auto", lineHeight: 1.7,
    fontFamily: "'Fira Code', 'Cascadia Code', monospace",
    margin: 0,
  }}>{code}</pre>
);

const strategies = [
  {
    id: "index-per-tenant",
    label: "Index Per Tenant",
    rec: "⚡ Best for < 500 users or strict isolation",
    pros: ["Full data isolation", "Easy tenant deletion", "Per-tenant tuning", "Simple ACL"],
    cons: ["Index count limits (~1000)", "Higher overhead", "Harder to query across tenants"],
    color: colors.green,
    code: `# Each tenant gets their own index
# tenant_id → opensearch index name

def get_index_name(tenant_id: str) -> str:
    return f"kb_{tenant_id}"

# User A's index: kb_user_a
# User B's index: kb_user_b

vectorstore = OpenSearchVectorSearch(
    index_name=get_index_name(tenant_id),
    embedding_function=embeddings,
    opensearch_url=OPENSEARCH_URL,
)

# Ingest only goes to their index
# Query only hits their index → zero bleed`,
  },
  {
    id: "shared-index",
    label: "Shared Index + Filter",
    rec: "🚀 Best for > 500 users or SaaS scale",
    pros: ["Scales to millions of users", "Low overhead", "Easy cross-tenant analytics", "Simple ops"],
    cons: ["Tenant isolation via query (not physical)", "Must enforce filter on EVERY query", "Harder per-tenant tuning"],
    color: colors.accent,
    code: `# All tenants share one index, filtered by tenant_id
INDEX_NAME = "knowledge_base_v1"

# Every document has tenant_id metadata
doc_metadata = {
    "tenant_id": tenant_id,
    "source": filename,
    "created_at": timestamp,
}

# Query ALWAYS filters by tenant_id (pre-filter)
def build_query(tenant_id, query_vector, k=5):
    return {
        "size": k,
        "query": {
            "bool": {
                "filter": [{"term": {"metadata.tenant_id": tenant_id}}],
                "must": [{"knn": {"vector_field": {
                    "vector": query_vector, "k": k
                }}}]
            }
        }
    }`,
  },
  {
    id: "hybrid",
    label: "Hybrid (Recommended)",
    rec: "🏆 Best for production multi-tenant SaaS",
    pros: ["Tier-based isolation", "Scale + security", "Enterprise = own index", "Cost efficient"],
    cons: ["More complex routing logic", "Two patterns to maintain"],
    color: colors.purple,
    code: `# Route based on tenant tier
def get_vector_store(tenant: Tenant) -> OpenSearchVectorSearch:
    if tenant.tier == "enterprise":
        # Dedicated index for enterprise customers
        index = f"kb_enterprise_{tenant.id}"
    else:
        # Shared pool for free/pro users
        index = f"kb_shared_pool_{tenant.region}"

    return OpenSearchVectorSearch(
        index_name=index,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL,
        http_auth=(OS_USER, OS_PASS),
    )

# Enterprise: index isolation
# Pro/Free: shared index + tenant_id filter`,
  },
];

export default function App() {
  const [activeStrategy, setActiveStrategy] = useState("hybrid");
  const [activeTab, setActiveTab] = useState("architecture");
  const strat = strategies.find(s => s.id === activeStrategy);

  return (
    <div style={{ background: colors.bg, minHeight: "100vh", color: colors.text, fontFamily: "'Inter', 'Segoe UI', sans-serif", padding: "24px 20px" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 32 }}>
        <div style={{ fontSize: 11, letterSpacing: 3, color: colors.accent, textTransform: "uppercase", marginBottom: 8 }}>
          End-to-End System Design
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 800, margin: 0, background: "linear-gradient(135deg, #60a5fa, #a78bfa)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Multi-Tenant RAG with OpenSearch
        </h1>
        <p style={{ color: colors.muted, fontSize: 13, marginTop: 8 }}>Python · LangChain · OpenSearch · FastAPI</p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 28 }}>
        {["architecture", "strategy", "flow"].map(tab => (
          <Tab key={tab} label={tab.charAt(0).toUpperCase() + tab.slice(1)} active={activeTab === tab} onClick={() => setActiveTab(tab)} />
        ))}
      </div>

      {/* ARCHITECTURE TAB */}
      {activeTab === "architecture" && (
        <div>
          {/* Tenants */}
          <div style={{ display: "flex", justifyContent: "center", gap: 16, marginBottom: 12, flexWrap: "wrap" }}>
            {[
              { name: "User A (Free)", color: colors.green },
              { name: "User B (Pro)", color: colors.accent },
              { name: "User C (Enterprise)", color: colors.purple },
            ].map(t => <TenantBadge key={t.name} {...t} />)}
          </div>

          <div style={{ display: "flex", justifyContent: "center", marginBottom: 4 }}>
            <Arrow vertical label="HTTP requests" />
          </div>

          {/* API Layer */}
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 4 }}>
            <Box color={colors.orange} title="FastAPI Gateway" icon="🔌" items={[
              "POST /ingest — Upload & chunk docs",
              "POST /query — RAG query endpoint",
              "JWT auth → extract tenant_id",
              "Rate limiting per tenant",
            ]} style={{ minWidth: 320 }} />
          </div>

          <div style={{ display: "flex", justifyContent: "center", marginBottom: 4 }}>
            <Arrow vertical label="routes to pipeline" />
          </div>

          {/* Two pipelines */}
          <div style={{ display: "flex", justifyContent: "center", gap: 24, flexWrap: "wrap", marginBottom: 4 }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <Box color={colors.green} title="Ingestion Pipeline" icon="📥" items={[
                "Doc loader (PDF/TXT/DOCX)",
                "Text splitter (RecursiveChar)",
                "Embeddings (OpenAI/Bedrock)",
                "Store → OpenSearch",
              ]} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <Box color={colors.accent} title="Inference Pipeline" icon="🤖" items={[
                "Embed user query",
                "kNN search (tenant filtered)",
                "Retrieve top-k chunks",
                "LLM answer (GPT/Claude)",
              ]} />
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "center", marginBottom: 4 }}>
            <Arrow vertical label="read / write vectors" />
          </div>

          {/* OpenSearch */}
          <div style={{ display: "flex", justifyContent: "center", gap: 16, flexWrap: "wrap", marginBottom: 4 }}>
            <Box color={colors.red} title="OpenSearch Cluster" icon="🔍" items={[
              "kNN plugin (FAISS/HNSW)",
              "Index: kb_shared_pool",
              "Index: kb_enterprise_xyz",
              "BM25 + vector hybrid",
            ]} style={{ minWidth: 300 }} />
            <Box color={colors.muted} title="Supporting Services" icon="⚙️" items={[
              "PostgreSQL — tenant metadata",
              "Redis — query cache",
              "S3 — raw document store",
              "Celery — async ingestion",
            ]} />
          </div>
        </div>
      )}

      {/* STRATEGY TAB */}
      {activeTab === "strategy" && (
        <div>
          <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 20, flexWrap: "wrap" }}>
            {strategies.map(s => (
              <button key={s.id} onClick={() => setActiveStrategy(s.id)} style={{
                background: activeStrategy === s.id ? `${s.color}22` : "transparent",
                border: `1.5px solid ${activeStrategy === s.id ? s.color : colors.border}`,
                color: activeStrategy === s.id ? s.color : colors.muted,
                borderRadius: 8, padding: "8px 16px", fontSize: 12,
                cursor: "pointer", fontFamily: "inherit",
              }}>{s.label}</button>
            ))}
          </div>

          <div style={{ maxWidth: 700, margin: "0 auto" }}>
            <div style={{ background: `${strat.color}15`, border: `1px solid ${strat.color}44`, borderRadius: 10, padding: "10px 16px", marginBottom: 16, fontSize: 13, color: strat.color }}>
              {strat.rec}
            </div>

            <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
              <div style={{ flex: 1, background: "#0d1f1599", border: "1px solid #10b98133", borderRadius: 8, padding: 14 }}>
                <div style={{ fontSize: 11, color: colors.green, fontWeight: 700, marginBottom: 8 }}>✅ PROS</div>
                {strat.pros.map((p, i) => <div key={i} style={{ fontSize: 12, color: colors.text, marginBottom: 4 }}>• {p}</div>)}
              </div>
              <div style={{ flex: 1, background: "#1f0d0d99", border: "1px solid #ef444433", borderRadius: 8, padding: 14 }}>
                <div style={{ fontSize: 11, color: colors.red, fontWeight: 700, marginBottom: 8 }}>⚠️ CONS</div>
                {strat.cons.map((c, i) => <div key={i} style={{ fontSize: 12, color: colors.text, marginBottom: 4 }}>• {c}</div>)}
              </div>
            </div>

            <CodeBlock code={strat.code} />
          </div>
        </div>
      )}

      {/* FLOW TAB */}
      {activeTab === "flow" && (
        <div style={{ maxWidth: 680, margin: "0 auto" }}>
          {[
            {
              phase: "1. Ingestion Flow", color: colors.green, icon: "📥",
              steps: [
                "User uploads file → POST /ingest {tenant_id, file}",
                "Load doc (PyPDFLoader, TextLoader, etc.)",
                "Split into chunks (RecursiveCharacterTextSplitter)",
                "Embed each chunk (OpenAIEmbeddings / BedrockEmbeddings)",
                "Store to OpenSearch with metadata: {tenant_id, source, chunk_id}",
                "Save doc record to PostgreSQL for management",
              ]
            },
            {
              phase: "2. Query / Inference Flow", color: colors.accent, icon: "🔎",
              steps: [
                "User sends query → POST /query {tenant_id, question}",
                "Check Redis cache (tenant_id + question hash)",
                "Embed the question using same embedding model",
                "kNN search on OpenSearch filtered by tenant_id",
                "Retrieve top-k chunks as context",
                "Feed context + question to LLM → stream answer",
              ]
            },
            {
              phase: "3. Security & Isolation", color: colors.purple, icon: "🔒",
              steps: [
                "JWT token → decode → extract tenant_id on every request",
                "tenant_id injected into ALL OpenSearch queries as pre-filter",
                "Middleware enforces tenant_id matches token — no override possible",
                "OpenSearch field-level security (optional, enterprise)",
                "Audit log: every query logged with tenant_id + timestamp",
              ]
            },
          ].map(({ phase, color, icon, steps }) => (
            <div key={phase} style={{ marginBottom: 20, border: `1px solid ${color}33`, borderRadius: 10, overflow: "hidden" }}>
              <div style={{ background: `${color}22`, padding: "10px 16px", fontSize: 13, fontWeight: 700, color, display: "flex", gap: 8 }}>
                <span>{icon}</span>{phase}
              </div>
              <div style={{ padding: "12px 16px" }}>
                {steps.map((s, i) => (
                  <div key={i} style={{ display: "flex", gap: 10, marginBottom: 8, alignItems: "flex-start" }}>
                    <div style={{ width: 20, height: 20, borderRadius: "50%", background: `${color}33`, border: `1px solid ${color}66`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color, flexShrink: 0, marginTop: 1 }}>{i + 1}</div>
                    <div style={{ fontSize: 12.5, color: colors.text, lineHeight: 1.5 }}>{s}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ textAlign: "center", marginTop: 28, fontSize: 11, color: colors.muted }}>
        Click tabs to explore Architecture · Strategy · Flow
      </div>
    </div>
  );
}
