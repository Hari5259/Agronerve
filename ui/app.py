import sys
import io
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image
from core.orchestrator import AgentOrchestrator, DOMAIN_CONFIGS
from core.knowledge_seeder import KnowledgeChunker
from core.vision_analyzer import leaf_vision_scanner
from core.sensor_telemetry import sensor_manager
from core.translator import language_manager, SUPPORTED_LANGUAGES
from core.voice_engine import voice_engine
from core.session_manager import session_manager
from evaluation.benchmark import AgroNerveBenchmark

st.set_page_config(
    page_title="AgroNerve - Multimodal Agricultural AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 1.3rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        color: #d8f3dc !important;
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .main-header p {
        color: #b7e4c7;
        margin: 0.2rem 0 0 0;
        font-size: 0.92rem;
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
    .badge-composite { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
    .badge-general { background-color: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; }
    .upload-box {
        background: #f0fdf4;
        border: 1.5px dashed #86efac;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "orchestrator" not in st.session_state or not hasattr(st.session_state.orchestrator, "process_multimodal_turn"):
    st.session_state.orchestrator = AgentOrchestrator()

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "en"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to AgroNerve Multimodal AI!**\n\nYou can **type any farming question** or **attach a picture of your crop leaf** (e.g. Tomato, Paddy, Cotton) below.\n\nI will diagnose the plant disease, quantify foliar damage %, provide verified ICAR treatments, and let you continue chatting with full conversation memory!",
            "meta": None,
            "image": None
        }
    ]

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    
    st.markdown("### 🌐 Regional Language")
    selected_lang_name = st.selectbox(
        "Choose Interface Language:",
        list(SUPPORTED_LANGUAGES.values()),
        index=0
    )
    old_lang = st.session_state.selected_lang
    for code, name in SUPPORTED_LANGUAGES.items():
        if name == selected_lang_name:
            st.session_state.selected_lang = code

    if old_lang != st.session_state.selected_lang:
        # If the only message is the welcome message, localize it dynamically
        if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
            if st.session_state.selected_lang == "en":
                st.session_state.messages[0]["content"] = "👋 **Welcome to AgroNerve Multimodal AI!**\n\nYou can **type any farming question** or **attach a picture of your crop leaf** (e.g. Tomato, Paddy, Cotton) below.\n\nI will diagnose the plant disease, quantify foliar damage %, provide verified ICAR treatments, and let you continue chatting with full conversation memory!"
            else:
                st.session_state.messages[0]["content"] = f"👋 **{language_manager.get_text('welcome_msg', st.session_state.selected_lang)}**"

    st.markdown("---")
    st.markdown("### 🧠 Active Chat Context")
    sess = session_manager.get_or_create_session(st.session_state.session_id)
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    st.write(f"🌱 **Crop:** {sess.current_crop or 'None detected yet'}")
    st.write(f"🔬 **Diagnosed:** {sess.current_diagnosed_disease or 'None'}")
    
    if st.button("🔄 Reset Conversation Memory", use_container_width=True):
        session_manager.clear_session(st.session_state.session_id)
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation memory reset! How can I assist you with your crops today?",
                "meta": None,
                "image": None
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Quick Sample Queries")
    sample_queries = [
        ("🌿 Paddy Blast Disease", "There are spindle-shaped brown spots with grey center on my paddy leaves"),
        ("🧪 Stem Borer Dosage", "What is the recommended dosage of Chlorantraniliprole 18.5 SC for stem borer in paddy?"),
        ("🌦️ Spray Rain Window", "Should I spray pesticide if heavy rainfall is forecasted tomorrow?"),
        ("💧 Tomato Water Schedule", "How many days interval should I water tomato plants during flowering stage?"),
        ("⚡ Compound: Wilt + Rain", "My tomato crop is wilting, should I irrigate or wait for rain tomorrow?")
    ]

    selected_sample = None
    for label, query in sample_queries:
        if st.button(label, use_container_width=True):
            selected_sample = query

    st.markdown("---")
    st.caption("AgroNerve v1.1.0 • Multimodal Edge AI Advisory")

current_lang = st.session_state.selected_lang

# Header Banner
st.markdown(f"""
<div class="main-header">
    <h1>🌾 {language_manager.get_text('app_title', current_lang)}</h1>
    <p>{language_manager.get_text('tagline', current_lang)}</p>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_chat, tab_scan, tab_sensor, tab_kb, tab_calc, tab_bench = st.tabs([
    "💬 " + language_manager.get_text('chat_tab', current_lang),
    "📸 " + language_manager.get_text('scan_tab', current_lang),
    "📡 " + language_manager.get_text('sensor_tab', current_lang),
    "📚 " + language_manager.get_text('kb_tab', current_lang),
    "🧮 " + language_manager.get_text('calc_tab', current_lang),
    "📊 " + language_manager.get_text('bench_tab', current_lang)
])

# ----------------- TAB 1: ADVISORY CHAT WITH MULTIMODAL VISION -----------------
with tab_chat:
    # Photo attachment expander directly in Chat
    with st.expander("📷 **Attach / Capture Crop Leaf Photo for AI Diagnosis**", expanded=False):
        col_u1, col_u2 = st.columns([3, 2])
        with col_u1:
            uploaded_chat_photo = st.file_uploader(
                "Upload a leaf image (Tomato, Paddy, Cotton, Wheat, Chilli):", 
                type=["jpg", "jpeg", "png"],
                key="chat_uploader"
            )
        with col_u2:
            chat_crop_hint = st.selectbox(
                "Crop Type:", 
                ["Auto-detect", "Tomato", "Paddy (Rice)", "Cotton", "Wheat", "Chilli"],
                key="chat_crop_select"
            )

        photo_query = st.text_input("Optional question about this leaf image:", placeholder="e.g., What disease is this and how do I cure it?")
        if uploaded_chat_photo and st.button("🚀 Analyze Leaf Image & Send to AI Chat", key="btn_send_photo"):
            photo_bytes = uploaded_chat_photo.read()
            with st.spinner("Analyzing foliar visual patterns & retrieving ICAR management protocols..."):
                mm_res = st.session_state.orchestrator.process_multimodal_turn(
                    image_bytes=photo_bytes,
                    user_text=photo_query or f"Please analyze this leaf photograph of {chat_crop_hint if chat_crop_hint != 'Auto-detect' else 'my crop'} and suggest cure.",
                    session_id=st.session_state.session_id,
                    crop_hint=chat_crop_hint if chat_crop_hint != "Auto-detect" else "auto",
                    language=current_lang
                )
                
                # Append user image message
                st.session_state.messages.append({
                    "role": "user",
                    "content": photo_query or "Uploaded crop leaf image for diagnosis.",
                    "meta": {"has_image": True},
                    "image": photo_bytes
                })
                # Append assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": mm_res["response"],
                    "meta": mm_res,
                    "image": None
                })
                st.rerun()

    # Render chat message stream
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image"):
                st.image(msg["image"], caption="Attached Leaf Photo", width=250)

            if msg.get("meta") and msg["role"] == "assistant":
                meta = msg["meta"]
                domain = meta.get("domain", "general")
                is_multi = meta.get("is_multi_domain", False)
                agent_name = meta.get("agent_name", "Specialist Agent")
                latency = meta.get("latency_seconds", 0.0)
                engine = meta.get("engine", "Offline Engine")
                
                badge_class = "badge-composite" if is_multi else (f"badge-{domain}" if domain in ["disease", "pesticide", "weather", "irrigation"] else "badge-general")
                st.markdown(
                    f'<span class="domain-badge {badge_class}">⚡ {agent_name} • {latency}s [{engine}]</span>', 
                    unsafe_allow_html=True
                )

            st.markdown(msg["content"])

            # Voice Speech Readout Component
            if msg["role"] == "assistant" and len(msg["content"]) > 10:
                speech_text = voice_engine.clean_text_for_speech(msg["content"])
                st.components.v1.html(
                    voice_engine.generate_html5_audio_speech_script(speech_text, lang=current_lang),
                    height=45
                )

            if msg.get("meta") and msg["meta"].get("context_preview"):
                with st.expander("🔍 View Grounding Knowledge Chunks (RAG Context)"):
                    st.text(msg["meta"]["context_preview"])

    user_input = st.chat_input(language_manager.get_text('input_placeholder', current_lang))
    prompt_to_process = selected_sample or user_input

    if prompt_to_process:
        st.session_state.messages.append({"role": "user", "content": prompt_to_process, "meta": None, "image": None})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("Thinking & generating agricultural advisory..."):
                res = st.session_state.orchestrator.process_query(
                    prompt_to_process, 
                    session_id=st.session_state.session_id,
                    language=current_lang
                )

                domain = res["domain"]
                is_multi = res.get("is_multi_domain", False)
                agent_name = res["agent_name"]
                latency = res["latency_seconds"]
                engine = res["engine"]
                badge_class = "badge-composite" if is_multi else (f"badge-{domain}" if domain in ["disease", "pesticide", "weather", "irrigation"] else "badge-general")

                st.markdown(
                    f'<span class="domain-badge {badge_class}">⚡ {agent_name} • {latency}s [{engine}]</span>', 
                    unsafe_allow_html=True
                )
                st.markdown(res["response"])

                speech_text = voice_engine.clean_text_for_speech(res["response"])
                st.components.v1.html(
                    voice_engine.generate_html5_audio_speech_script(speech_text, lang=current_lang),
                    height=45
                )

                with st.expander("🔍 View Grounding Knowledge Chunks (RAG Context)"):
                    st.text(res["context_preview"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["response"],
                    "meta": res,
                    "image": None
                })

# ----------------- TAB 2: VISUAL LEAF SCANNER -----------------
with tab_scan:
    st.subheader("📸 Offline Leaf Disease Visual Diagnostic Scanner")
    st.write("Upload or capture a photograph of an affected crop leaf to analyze chlorosis, necrotic lesion density, and obtain immediate verified treatments.")

    uploaded_img = st.file_uploader("Upload Leaf Photo (JPG/PNG):", type=["jpg", "jpeg", "png"], key="scanner_tab_upload")
    col_crop, col_scan = st.columns([2, 1])
    with col_crop:
        crop_hint = st.selectbox("Select Crop Type (optional):", ["Auto-detect", "Paddy (Rice)", "Tomato", "Cotton", "Wheat", "Chilli"])

    if uploaded_img is not None:
        c_left, c_right = st.columns([1, 1])
        img_bytes = uploaded_img.read()
        with c_left:
            st.image(img_bytes, caption="Uploaded Leaf Image", use_container_width=True)

        with c_right:
            with st.spinner("Scanning leaf visual patterns & lesion density..."):
                scan_res = leaf_vision_scanner.analyze_image_bytes(img_bytes, crop_hint=crop_hint)
                
                if scan_res.get("status") == "success":
                    st.success(f"### 🔬 Diagnosis: **{scan_res['predicted_disease']}**")
                    st.metric("Visual Confidence", f"{scan_res['confidence_pct']}%")
                    st.metric("Estimated Affected Leaf Area", f"{scan_res['affected_leaf_area_pct']}%")

                    st.markdown("#### 📊 Foliar Surface Damage Breakdown")
                    metrics = scan_res.get("metrics", {})
                    st.progress(metrics.get("chlorosis_yellow_pct", 0) / 100.0, text=f"Chlorosis (Yellowing): {metrics.get('chlorosis_yellow_pct')}%")
                    st.progress(metrics.get("necrotic_brown_pct", 0) / 100.0, text=f"Necrotic Lesions: {metrics.get('necrotic_brown_pct')}%")
                    st.progress(metrics.get("dark_lesion_pct", 0) / 100.0, text=f"Dark Blight Spots: {metrics.get('dark_lesion_pct')}%")

                    st.markdown("#### 💊 Verified ICAR Management Protocol")
                    st.info(scan_res.get("verified_protocol", ""))
                else:
                    st.error(scan_res.get("message", "Error scanning leaf."))

# ----------------- TAB 3: IOT SENSOR TELEMETRY -----------------
with tab_sensor:
    st.subheader("📡 Live Soil & Environmental Sensor Telemetry")
    st.write("Real-time telemetry integration with on-device / Bluetooth soil probes and meteorological sensors.")

    t = sensor_manager.get_telemetry()
    soil = t["soil"]
    env = t["environment"]
    adv = t["agronomic_advisories"]

    st.caption(f"Last Sensor Sync: `{t['timestamp']}`")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🌱 Soil Moisture (VWC)", f"{soil['moisture_vwc_pct']}%", delta="Normal Hydration" if soil["status"] == "NORMAL" else "Alert")
    s2.metric("🌡️ Soil Temperature", f"{soil['temperature_celsius']} °C")
    s3.metric("☀️ Ambient Temp", f"{env['ambient_temp_celsius']} °C")
    s4.metric("💧 Relative Humidity", f"{env['relative_humidity_pct']}%")

    st.markdown("---")
    st.markdown("#### 🚜 Dynamic Field Action Triggers")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.info(f"**💧 Irrigation Trigger:**\n\n{adv['irrigation_action']}")
    with col_a2:
        st.warning(f"**🦠 Pathogen Infection Risk:**\n\n{adv['pathogen_infection_risk']}")

    st.success(f"**🧪 Chemical Spray Window Status:** {adv['chemical_spray_window']}")

# ----------------- TAB 4: KNOWLEDGE BASE -----------------
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

# ----------------- TAB 5: AGRO-CALCULATORS -----------------
with tab_calc:
    st.subheader("🧮 " + language_manager.get_text('calc_tab', current_lang))
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"#### 🧪 {language_manager.get_text('dosage_calc', current_lang)}")
        tank_size = st.number_input(language_manager.get_text('tank_size', current_lang), min_value=1.0, value=16.0, step=1.0)
        dosage_rate = st.number_input(language_manager.get_text('dosage_rate', current_lang), min_value=0.1, value=0.5, step=0.1)
        total_chemical = tank_size * dosage_rate
        st.success(f"**{language_manager.get_text('calc_result', current_lang)}:** `{total_chemical:.2f} ml (or grams)` per tank.")

    with c2:
        st.markdown(f"#### 💧 {language_manager.get_text('water_calc', current_lang)}")
        crop_selected = st.selectbox("Crop Type:", ["Paddy (Rice)", "Tomato", "Cotton", "Wheat"])
        soil_selected = st.selectbox("Soil Texture:", ["Clay / Black Cotton", "Loam", "Sandy Loam"])
        area_acres = st.number_input("Field Area (Acres):", min_value=0.1, value=1.0, step=0.5)
        
        base_req = {"Paddy (Rice)": 1200, "Tomato": 500, "Cotton": 700, "Wheat": 400}
        mm_req = base_req.get(crop_selected, 500)
        total_m3 = (mm_req / 1000.0) * (area_acres * 4046.86)
        st.info(f"**Total Seasonal Requirement:** ~`{mm_req} mm` (~`{total_m3:,.0f} m³` for {area_acres} acre(s))")

# ----------------- TAB 6: BENCHMARKS -----------------
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
