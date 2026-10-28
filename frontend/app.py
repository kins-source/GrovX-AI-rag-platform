import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from agents.llm_provider import get_nvidia_models

load_dotenv()

PROVIDER_API_KEY_ENVS = {
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
    "grok": ("XAI_API_KEY",),
    "nvidia": ("NVIDIA_NIM_API_KEY", "NVIDIA_NIMS_API_KEY", "NVIDIA_API_KEY"),
}


def provider_env_api_key(provider):
    for variable in PROVIDER_API_KEY_ENVS.get(provider, ()):
        value = os.getenv(variable, "").strip()
        if value and not value.startswith("process.env."):
            return value
    return ""


def default_provider():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider != "ollama":
        return provider
    for candidate in ("gemini", "nvidia", "openai", "openrouter", "grok"):
        if provider_env_api_key(candidate):
            return candidate
    return provider


API_KEY = os.getenv("API_KEY", "default-dev-key")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-KEY": API_KEY}

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Enterprise AI Engine", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="collapsed" # We collapse default just in case CSS doesn't hide it instantly
)

# ----------------- SESSION STATE -----------------
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploaded_outputs" not in st.session_state:
    st.session_state.uploaded_outputs = {}
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "avg_latency" not in st.session_state:
    st.session_state.avg_latency = 0.0
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = default_provider()
if "applied_provider" not in st.session_state:
    st.session_state.applied_provider = st.session_state.llm_provider
if "applied_api_key" not in st.session_state:
    st.session_state.applied_api_key = provider_env_api_key(st.session_state.applied_provider)
if "editing_api_key" not in st.session_state:
    st.session_state.editing_api_key = False
if "nvidia_model" not in st.session_state:
    st.session_state.nvidia_model = os.getenv("NVIDIA_MODEL", get_nvidia_models()[0])
if "applied_nvidia_model" not in st.session_state:
    st.session_state.applied_nvidia_model = st.session_state.nvidia_model

# ----------------- GLOBAL CSS -----------------
st.markdown("""
<style>
    /* SaaS Global Styling */
    .stApp { background: radial-gradient(circle at 88% 0%, #dbeafe 0%, #f7fbff 32%, #eef4f9 100%); color: #0b1220; }
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Completely hide native sidebar and toggle buttons */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1000px;
    }
    [data-testid="stMetric"] { background: rgba(255,255,255,0.78); border: 1px solid #cbdced; border-radius: 16px; padding: 1rem; box-shadow: 0 10px 28px rgba(15, 35, 61, 0.08); }
    .stButton > button { border-radius: 12px; border: 1px solid #b5c8dc; background: #ffffff; color: #0f2742; font-weight: 650; transition: all 0.2s ease; }
    .stButton > button:hover { border-color: #2563a8; background: #e8f2ff; color: #0b3c70; transform: translateY(-1px); }
    [data-testid="stChatInput"] { border: 1px solid #a9bfd6; border-radius: 16px; background: rgba(255,255,255,0.92); box-shadow: 0 12px 30px rgba(15, 35, 61, 0.1); box-sizing: border-box; }
    [data-testid="stChatInput"] > div { width: 100% !important; box-sizing: border-box; }
    [data-testid="stChatInput"] textarea { padding-right: 3.5rem !important; }
    [data-testid="stChatInput"] button { display: flex !important; align-items: center; justify-content: center; width: 2.75rem !important; min-width: 2.75rem !important; height: 2.75rem !important; flex-shrink: 0 !important; visibility: visible !important; opacity: 1 !important; z-index: 2; color: #0f2742 !important; background: #e8f2ff !important; }
    [data-testid="stChatInput"] button svg { width: 1.25rem !important; height: 1.25rem !important; }

    /* Modern Chat Bubble Settings */
    /* AI Message Bubble */
    [data-testid="stChatMessage"] {
        background-color: #f9fafb !important;
        border: 1px solid #eaebec;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    /* User Message Bubble */
    [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(odd) {
        background-color: #e5e7eb !important;
        border: 1px solid #d1d5db;
    }
    /* Force text visibility and contrast */
    [data-testid="stChatMessage"] * {
        color: #111827 !important;
        opacity: 1 !important;
    }
    .stChatInput { border-radius: 20px; }

    /* Custom Sidebar Styling using CSS :has() */
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) {
        position: fixed !important;
        top: 0;
        left: 0;
        width: 280px;
        height: 100vh;
        background: linear-gradient(160deg, #07111f 0%, #0c1f35 55%, #102f4e 100%);
        z-index: 999999;
        padding: 2rem 1.5rem;
        overflow-y: auto;
        transition: transform 0.3s ease-in-out;
        border-right: 1px solid #1e293b;
        box-shadow: 12px 0 32px rgba(2, 12, 25, 0.3);
        border-radius: 0 18px 18px 0;
    }

    /* Force all text inside custom sidebar to white */
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) * {
        color: #ffffff !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) [data-testid="stFileUploader"] * {
        color: #111827 !important;
    }
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) [data-testid="stFileUploader"] button,
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    /* Custom button styling within dark sidebar */
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) button {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
    }
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) button:hover {
        border: 1px solid #60a5fa !important;
        background-color: #173b60 !important;
    }
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) input,
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) [data-baseweb="select"] > div {
        border-radius: 10px !important;
        background-color: #132943 !important;
        border-color: #345777 !important;
    }
    div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) hr { opacity: 0.45; }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR TOGGLE LOGIC -----------------
if st.session_state.sidebar_open:
    st.markdown("""
        <style>
            div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) {
                transform: translateX(0);
            }
            .block-container {
                padding-left: 320px !important;
                transition: padding-left 0.3s ease-in-out;
            }
            [data-testid="stChatInput"] {
                left: 320px !important;
                right: auto !important;
                width: calc(100% - 340px) !important;
                transform: none !important;
                transition: left 0.3s ease-in-out, right 0.3s ease-in-out;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            div[data-testid="stVerticalBlock"]:has(> div.element-container div#my-custom-sidebar) {
                transform: translateX(-100%);
            }
            .block-container {
                padding-left: 3rem !important;
                transition: padding-left 0.3s ease-in-out;
            }
            [data-testid="stChatInput"] {
                left: 3rem !important;
                right: auto !important;
                width: calc(100% - 6rem) !important;
                transform: none !important;
                transition: left 0.3s ease-in-out, right 0.3s ease-in-out;
            }
        </style>
    """, unsafe_allow_html=True)

# ----------------- CUSTOM SIDEBAR CONTAINER -----------------
sidebar_container = st.container()
with sidebar_container:
    # Invisible marker to anchor CSS Targeting
    st.markdown('<div id="my-custom-sidebar" style="display:none;"></div>', unsafe_allow_html=True)
    
    st.markdown("<h2 style='margin-bottom:0;'>⚡ AI Workspace</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #1e293b; margin-top: 1rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    
    # 1. Documents Section
    st.markdown("<h4>📂 Documents</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Data",
        type=["pdf", "txt", "md", "markdown", "csv", "doc", "docx", "ppt", "pptx", "xls", "xlsx"],
        label_visibility="collapsed",
    )
    if st.button("Upload & Index", use_container_width=True):
        if uploaded_file:
            with st.spinner("Processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, headers=HEADERS)
                    if response.status_code == 200:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        upload_data = response.json()
                        st.session_state.uploaded_outputs[uploaded_file.name] = upload_data
                        st.success(upload_data.get("message", "Indexed!"))
                        with st.expander("Uploaded file output", expanded=True):
                            st.caption(f"{upload_data.get('pages', 0)} sections, {upload_data.get('characters', 0)} characters")
                            st.text(upload_data.get("preview", "No text was extracted."))
                    else:
                        try:
                            detail = response.json().get("detail", response.text)
                        except ValueError:
                            detail = response.text
                        st.error(f"Upload failed: {detail or 'No details returned.'}")
                except Exception as e:
                    st.error(f"Backend error: {e}")
    
    if st.session_state.uploaded_files:
        st.markdown("<p style='font-size: 0.9em; margin-top:1em; margin-bottom:0;'>Indexed Files:</p>", unsafe_allow_html=True)
        for f in set(st.session_state.uploaded_files):
            st.caption(f"📄 {f}")
            output = st.session_state.uploaded_outputs.get(f)
            if output:
                with st.expander(f"View output: {f}"):
                    st.text(output.get("preview", "No text was extracted."))
            
    st.markdown("<hr style='border-color: #1e293b; margin-top: 1.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    
    # 2. History Section
    st.markdown("<h4>🕘 Query History</h4>", unsafe_allow_html=True)
    user_message_indices = [
        index for index, message in enumerate(st.session_state.messages)
        if message["role"] == "user"
    ]
    if user_message_indices:
        for message_index in reversed(user_message_indices[-5:]):
            message = st.session_state.messages[message_index]
            original_query = message["content"]
            history_title = message.get("history_title", original_query)
            history_cols = st.columns([0.58, 0.21, 0.21], gap="small")
            with history_cols[0]:
                renamed_title = st.text_input(
                    "History item",
                    value=history_title,
                    key=f"history_title_{message_index}",
                    label_visibility="collapsed",
                )
            with history_cols[1]:
                if st.button("Rename", key=f"rename_history_{message_index}", help="Rename history item", use_container_width=True):
                    message["history_title"] = renamed_title.strip() or original_query
                    st.rerun()
            with history_cols[2]:
                if st.button("Delete", key=f"delete_history_{message_index}", help="Delete history item", use_container_width=True):
                    del st.session_state.messages[message_index]
                    if message_index < len(st.session_state.messages) and st.session_state.messages[message_index]["role"] == "assistant":
                        del st.session_state.messages[message_index]
                    st.session_state.query_count = sum(
                        1 for item in st.session_state.messages
                        if item["role"] == "assistant" and "latency" in item
                    )
                    st.rerun()
    else:
        st.caption("No queries run yet.")

    st.markdown("<hr style='border-color: #1e293b; margin-top: 1.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    
    # 3. Settings Section
    st.markdown("<h4>⚙️ Settings</h4>", unsafe_allow_html=True)
    provider_names = {
        "ollama": "Ollama",
        "openai": "ChatGPT / OpenAI",
        "openrouter": "OpenRouter",
        "gemini": "Google Gemini",
        "grok": "xAI Grok",
        "nvidia": "NVIDIA NIM",
    }
    provider = st.selectbox(
        "LLM Provider",
        list(provider_names),
        format_func=lambda value: provider_names[value],
        key="llm_provider",
    )
    if provider == "ollama":
        st.caption("Uses the configured Ollama server. No API key required.")
    else:
        if provider == "nvidia":
            st.selectbox(
                "NVIDIA NIM model",
                get_nvidia_models(),
                key="nvidia_model",
            )
        configured_key = provider_env_api_key(provider)
        if st.session_state.applied_provider == provider and st.session_state.applied_api_key and not st.session_state.editing_api_key:
            st.caption(f"{provider_names[provider]} API key is active.")
            if st.button("Change API key", key="change_api_key"):
                st.session_state.editing_api_key = True
                st.rerun()
        else:
            st.text_input(
                f"{provider_names[provider]} API key",
                type="password",
                key=f"llm_api_key_{provider}",
                placeholder="Optional: paste a key to override .env",
            )
            if configured_key:
                st.caption("Using the API key configured in .env unless you enter an override.")
    if st.button("Apply API Key", use_container_width=True, key="apply_api_key"):
        typed_key = st.session_state.get(f"llm_api_key_{provider}", "").strip()
        resolved_key = typed_key or provider_env_api_key(provider)
        if provider != "ollama" and not resolved_key:
            st.error("Enter an API key before applying this provider.")
        else:
            st.session_state.applied_provider = provider
            st.session_state.applied_api_key = resolved_key
            st.session_state.editing_api_key = False
            if provider == "nvidia":
                st.session_state.applied_nvidia_model = st.session_state.nvidia_model
            st.success(f"{provider_names[provider]} is ready for queries.")
    if st.session_state.applied_provider == provider:
        st.caption(f"Active provider: {provider_names[provider]}")


# ----------------- NAVBAR -----------------
with st.container():
    col_btn, col_title = st.columns([0.08, 0.92])

    with col_btn:
        if st.button("☰", key="hamburger", help="Toggle workspace panel"):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()

    with col_title:
        st.markdown("""
        <h1 style="
            color: #111827;
            font-weight: 700;
            font-size: 40px;
            margin-bottom: 5px;
        ">
        ⚡ Enterprise AI Knowledge Assistant
        </h1>
        """, unsafe_allow_html=True)

        st.markdown("<p style='color:#6c757d; font-size:1.1rem; margin-bottom:2rem;'>Securely query organizational metrics, sales records, and compliance requirements.</p>", unsafe_allow_html=True)
# ----------------- DASHBOARD METRICS -----------------
col1, col2, col3 = st.columns(3)
m_col1 = col1.empty()
m_col2 = col2.empty()
m_col3 = col3.empty()

def render_metrics():
    # Render into empty placeholders so they stay at the top
    with m_col1:
        st.metric("Queries Run", st.session_state.query_count)
    with m_col2:
        st.metric("Knowledge Objects", len(set(st.session_state.uploaded_files)))
    with m_col3:
        st.metric("Avg Latency", f"{st.session_state.avg_latency:.2f}s" if st.session_state.query_count > 0 else "0.00s")

render_metrics()
st.markdown("<br>", unsafe_allow_html=True)

# ----------------- EMPTY STATE UI -----------------
prompt = None

if len(st.session_state.messages) == 0:
    st.info("👋 Welcome to the Intelligence Center. Connect to your data to begin.")
    st.markdown("<h4 style='color:#495057; font-size: 1rem; margin-top:20px;'>✨ Try asking:</h4>", unsafe_allow_html=True)
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    if btn_col1.button("What is total revenue?", use_container_width=True):
        prompt = "What is total revenue?"
    if btn_col2.button("Summarize uploaded document", use_container_width=True):
        prompt = "Summarize uploaded document"
    if btn_col3.button("Show compliance insights", use_container_width=True):
        prompt = "Show compliance insights"

# ----------------- CHAT INTERFACE -----------------
# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "latency" in msg:
            st.caption(f"⏱️ {msg['latency']}s | 📚 {len(msg.get('sources', []))} Citations")
            if msg.get("sources"):
                with st.expander("Verify Data Sources"):
                    for s in msg["sources"]:
                        st.write(f"- {s}")

# Handle main chat input bar
if chat_val := st.chat_input("Ask something about your data..."):
    prompt = chat_val

# ----------------- EXECUTION PIPELINE -----------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing parameters..."):
            try:
                payload = {
                    "query": prompt,
                    "provider": st.session_state.applied_provider,
                    "api_key": st.session_state.applied_api_key or None,
                    "model": st.session_state.applied_nvidia_model if st.session_state.applied_provider == "nvidia" else None,
                }
                response = requests.post(f"{BACKEND_URL}/query", json=payload, headers=HEADERS)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    
                    raw_latency = data.get("latency_seconds", 0)
                    latency = round(raw_latency, 2) if isinstance(raw_latency, (int, float)) else 0.0
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    st.caption(f"⏱️ {latency}s | 📚 {len(sources)} Citations")
                    if sources:
                        with st.expander("Verify Data Sources"):
                            for s in sources:
                                st.write(f"- {s}")
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "latency": latency,
                        "sources": sources
                    })
                    
                    # Update Metrics
                    current_total = st.session_state.avg_latency * st.session_state.query_count
                    st.session_state.query_count += 1
                    st.session_state.avg_latency = (current_total + latency) / st.session_state.query_count
                    
                    # Target metrics immediately
                    render_metrics()
                    
                    if prompt != chat_val:
                        st.rerun()

                else:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    error_msg = f"API error ({response.status_code}): {detail or 'No details returned.'}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Request failed: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
