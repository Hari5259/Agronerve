import streamlit as st
from core.orchestrator import AgentOrchestrator

st.set_page_config(
    page_title="AgroNerve - Agricultural Advisory",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 AgroNerve")
st.caption("Intelligent Offline Agricultural Advisory System")

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to AgroNerve! How can I assist you with your crops today? (e.g., disease diagnosis, pesticide dosage, weather planning, or irrigation schedules)"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a farming question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and assembling expert agent..."):
            result = st.session_state.orchestrator.process_query(prompt)
            domain_badge = f"**[Routed Domain: {result['domain'].capitalize()}]**\n\n"
            response_text = f"{domain_badge}Processing query in offline mode. Domain specialist agent has been assembled."
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
