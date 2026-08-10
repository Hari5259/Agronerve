import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from core.orchestrator import AgentOrchestrator, DOMAIN_CONFIGS
from core.knowledge_seeder import KnowledgeChunker
from evaluation.benchmark import AgroNerveBenchmark

st.set_page_config(
    page_title="AgroNerve - Offline Agricultural Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        color: #d8f3dc !important;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #b7e4c7;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .domain-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-disease { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .badge-pesticide { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-weather { background-color: #e0f2fe; color: #075985; border: 1px solid #bae6fd; }
    .badge-irrigation { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .badge-general { background-color: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; }
    .stat-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to AgroNerve!**\n\nI am your intelligent, offline agricultural advisor. I automatically detect your query intent and dynamically assemble a domain specialist agent for:\n- 🌿 **Crop Disease Diagnosis** (ICAR guidelines)\n- 🧪 **Pesticide & Dosage Safety** (CIBRC standards)\n- 🌦️ **Weather-Aware Farming** (Cached forecasts)\n- 💧 **Irrigation Scheduling** (FAO-56 models)\n\nHow can I help your farm today?",
            "meta": None
        }
    ]

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>🌾 AgroNerve</h1>
    <p>Offline Agricultural Advisory System with Dynamic Agent Orchestration & RAG</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### ⚙️ System Status")
    
    total_chunks = len(KnowledgeChunker.get_all_chunks())
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Knowledge Chunks", value=total_chunks)
    with c2:
        st.metric(label="Offline Status", value="Ready 🟢")

    st.markdown("---")
    st.markdown("### 💡 Quick Sample Queries")
    sample_queries = [
        ("🌿 Paddy Blast Disease", "There are spindle-shaped brown spots with grey center on my paddy leaves"),
        ("🧪 Stem Borer Dosage", "What is the recommended dosage of Chlorantraniliprole 18.5 SC for stem borer in paddy?"),
        ("🌦️ Spray Rain Window", "Should I spray pesticide if heavy rainfall is forecasted tomorrow?"),
        ("💧 Tomato Water Schedule", "How many days interval should I water tomato plants during flowering stage?")
    ]

    selected_sample = None
    for label, query in sample_queries:
        if st.button(label, use_container_width=True):
            selected_sample = query

    st.markdown("---")
    st.caption("AgroNerve v1.0.0 • 100% Offline Edge Advisory")

# Tabs for Navigation
tab_chat, tab_kb, tab_calc, tab_bench = st.tabs([
    "💬 Advisory Chat", 
    "📚 Knowledge Base", 
    "🧮 Agro-Calculators",
    "📊 Benchmarks"
])

with tab_chat:
    # Render chat message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("meta"):
                meta = msg["meta"]
                domain = meta.get("domain", "general")
                agent_name = meta.get("agent_name", "Specialist Agent")
                latency = meta.get("latency_seconds", 0.0)
                engine = meta.get("engine", "Offline Engine")
                
                badge_class = f"badge-{domain}" if domain in ["disease", "pesticide", "weather", "irrigation"] else "badge-general"
                st.markdown(
                    f'<span class="domain-badge {badge_class}">⚡ {agent_name} ({domain.upper()}) • {latency}s [{engine}]</span>', 
                    unsafe_allow_html=True
                )

            st.markdown(msg["content"])

            if msg.get("meta") and msg["meta"].get("context_preview"):
                with st.expander("🔍 View Grounding Knowledge Chunks (RAG Context)"):
                    st.text(msg["meta"]["context_preview"])

    # Chat Input handler
    user_input = st.chat_input("Describe your crop symptom, chemical dosage query, or weather question...")
    
    # If user clicked a sample button, use that
    prompt_to_process = selected_sample or user_input

    if prompt_to_process:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process, "meta": None})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Process Turn
        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent, retrieving ICAR/FAO knowledge, and assembling agent..."):
                res = st.session_state.orchestrator.process_query(prompt_to_process)

                domain = res["domain"]
                agent_name = res["agent_name"]
                latency = res["latency_seconds"]
                engine = res["engine"]
                badge_class = f"badge-{domain}" if domain in ["disease", "pesticide", "weather", "irrigation"] else "badge-general"

                st.markdown(
                    f'<span class="domain-badge {badge_class}">⚡ {agent_name} ({domain.upper()}) • {latency}s [{engine}]</span>', 
                    unsafe_allow_html=True
                )
                st.markdown(res["response"])

                with st.expander("🔍 View Grounding Knowledge Chunks (RAG Context)"):
                    st.text(res["context_preview"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["response"],
                    "meta": res
                })

with tab_kb:
    st.subheader("📚 Verified Agricultural Knowledge Corpus")
    st.write("Browse locally stored and embedded agricultural datasets sourced from ICAR, FAO, and CIBRC.")
    
    kb_domain = st.selectbox("Select Domain Partition:", ["All", "Disease", "Pesticide", "Weather", "Irrigation"])
    all_chunks = KnowledgeChunker.get_all_chunks()
    
    if kb_domain != "All":
        filtered = [c for c in all_chunks if c.get("domain") == kb_domain.lower()]
    else:
        filtered = all_chunks

    st.write(f"Showing **{len(filtered)}** verified entries:")
    for chunk in filtered:
        with st.expander(f"[{chunk.get('domain', '').upper()}] {chunk.get('title', 'Document')}"):
            st.code(chunk.get("text", ""), language="markdown")

with tab_calc:
    st.subheader("🧮 Offline Agronomic Utility Calculators")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🧪 Pesticide Spray Dilution Calculator")
        tank_size = st.number_input("Spray Tank Capacity (Liters):", min_value=1.0, value=16.0, step=1.0)
        dosage_rate = st.number_input("Recommended Dosage Rate (ml or g per Liter):", min_value=0.1, value=0.5, step=0.1)
        total_chemical = tank_size * dosage_rate
        st.success(f"**Required Chemical Quantity:** `{total_chemical:.2f} ml (or grams)` per tank.")

    with c2:
        st.markdown("#### 💧 Crop Water Requirement Estimator")
        crop_selected = st.selectbox("Crop Type:", ["Paddy (Rice)", "Tomato", "Cotton", "Wheat"])
        soil_selected = st.selectbox("Soil Texture:", ["Clay / Black Cotton", "Loam", "Sandy Loam"])
        area_acres = st.number_input("Field Area (Acres):", min_value=0.1, value=1.0, step=0.5)
        
        base_req = {"Paddy (Rice)": 1200, "Tomato": 500, "Cotton": 700, "Wheat": 400}
        mm_req = base_req.get(crop_selected, 500)
        total_m3 = (mm_req / 1000.0) * (area_acres * 4046.86)
        st.info(f"**Total Seasonal Requirement:** ~`{mm_req} mm` (~`{total_m3:,.0f} m³` for {area_acres} acre(s))")

with tab_bench:
    st.subheader("📊 AgroNerve Benchmark Suite")
    st.write("Evaluate two-stage intent routing accuracy and RAG recall over the 40-query test benchmark.")
    
    if st.button("🚀 Run Benchmark Evaluation"):
        with st.spinner("Running 40 test queries across 4 domains..."):
            runner = AgroNerveBenchmark()
            bench_res = runner.run_benchmark()
            
            b1, b2, b3 = st.columns(3)
            b1.metric("Overall Accuracy", f"{bench_res['overall_accuracy_pct']}%")
            b2.metric("Fast Keyword Rate", f"{bench_res['keyword_stage1_rate_pct']}%")
            b3.metric("Retrieval Recall@5", f"{bench_res['retrieval_recall_top5'] * 100}%")
            
            st.markdown("#### Domain Accuracies")
            st.json(bench_res["domain_accuracies"])
            
            st.markdown("#### Confusion Matrix")
            st.json(bench_res["confusion_matrix"])
