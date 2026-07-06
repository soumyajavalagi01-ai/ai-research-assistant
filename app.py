# app.py
import streamlit as st
from crew import run_research
from fpdf import FPDF
import datetime

st.set_page_config(
    page_title="AI Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Multi-Agent Research Assistant")
st.markdown("**Powered by:** CrewAI + Groq (LLaMA 3.3) + ChromaDB (RAG) + Tavily Search")
st.markdown("---")

with st.expander("⚙️ How This Works"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**Step 1**\n\n🔍 Tavily searches web for your topic")
    with col2:
        st.info("**Step 2**\n\n📥 Results stored in ChromaDB (RAG)")
    with col3:
        st.info("**Step 3**\n\n🤖 3 AI Agents analyze & write report")
    with col4:
        st.info("**Step 4**\n\n📄 Download report as PDF")

st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input(
        "🔎 Enter Research Topic:",
        placeholder="e.g., Artificial Intelligence in Healthcare 2026",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🚀 Start Research", type="primary", use_container_width=True)

if search_btn:
    if not topic.strip():
        st.warning("⚠️ Please enter a research topic!")
    else:
        with st.spinner("🤖 AI Agents are working... This takes 3-5 minutes..."):
            try:
                st.info("🚀 Starting research pipeline...")
                result = run_research(topic)

                # Status updates
                st.markdown("### 📋 Pipeline Status")
                for update in result["status_updates"]:
                    st.success(update)

                st.markdown("---")

                # Stats
                st.markdown("### 📊 Research Stats")
                m1, m2, m3 = st.columns(3)
                m1.metric("🌐 Web Sources", len(result["sources"]))
                m2.metric("📦 Chunks Stored (RAG)", result["chunks_stored"])
                m3.metric("🎯 Chunks Retrieved (RAG)", result["chunks_retrieved"])

                st.markdown("---")

                # Report
                st.markdown("### 📄 Research Report")
                st.markdown(result["report"])

                st.markdown("---")

                # Sources
                st.markdown("### 🔗 Sources Used")
                for i, url in enumerate(result["sources"], 1):
                    st.markdown(f"{i}. {url}")

                st.markdown("---")

                # PDF Download
                st.markdown("### 💾 Download Report")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, f"Research Report: {topic}", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                pdf.cell(0, 8, "Powered by: CrewAI + Google Gemini + ChromaDB RAG", ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", "", 11)
                clean_report = result["report"].encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 7, clean_report)
                pdf.ln(5)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Sources:", ln=True)
                pdf.set_font("Arial", "", 10)
                for url in result["sources"]:
                    clean_url = url.encode("latin-1", "replace").decode("latin-1")
                    pdf.cell(0, 7, clean_url, ln=True)
                pdf_output = bytes(pdf.output())

                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_output,
                    file_name=f"research_{topic[:30].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Check your API keys in the .env file")

st.markdown("---")
st.markdown("Built with ❤️ using **CrewAI** | **Groq LLaMA 3.3** | **ChromaDB (RAG)** | **Tavily Search** | **Streamlit**")