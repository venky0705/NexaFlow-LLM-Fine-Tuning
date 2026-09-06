from pathlib import Path
import sys
import time

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

APP_DIR = PROJECT_ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


# ============================================================
# IMPORT ENGINE
# ============================================================

from chatbot_engine import (
    load_chat_model,
    generate_response,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NexaFlow AI Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: #0f1117;
    color: #f3f4f6;
}

.block-container {
    max-width: 950px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}

section[data-testid="stSidebar"] {
    background: #17191f;
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.app-header {
    margin-bottom: 22px;
}

.app-title {
    font-size: 31px;
    font-weight: 750;
    letter-spacing: -0.7px;
    line-height: 1.2;
}

.app-subtitle {
    margin-top: 6px;
    color: #9ca3af;
    font-size: 14px;
}

.status-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 10px;
}

.status-label {
    color: #8b93a1;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-weight: 600;
}

.status-value {
    color: #f3f4f6;
    font-size: 14px;
    font-weight: 600;
    margin-top: 5px;
}

.empty-state {
    margin-top: 45px;
    text-align: center;
}

.empty-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 8px;
}

.empty-subtitle {
    color: #9ca3af;
    font-size: 14px;
    line-height: 1.6;
    max-width: 650px;
    margin: 0 auto 25px auto;
}

div[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 8px 4px;
    margin-bottom: 7px;
}

div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 18px;
    left: calc(50% + 110px);
    transform: translateX(-50%);
    width: min(900px, calc(100vw - 360px));
    z-index: 999;
}

.stButton > button {
    border-radius: 10px;
    min-height: 40px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.025);
    transition: all 0.15s ease-in-out;
}

.stButton > button:hover {
    border-color: rgba(255,255,255,0.22);
    background: rgba(255,255,255,0.06);
}

.response-meta {
    color: #747b88;
    font-size: 11px;
    margin-top: 5px;
}

.muted-text {
    color: #8b93a1;
    font-size: 12px;
}

@media (max-width: 900px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    div[data-testid="stChatInput"] {
        left: 50%;
        width: calc(100vw - 30px);
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CACHE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model():

    return load_chat_model()


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner(
    "Loading NexaFlow AI..."
):

    model, tokenizer = get_model()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = 1

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None


# ============================================================
# HELPERS
# ============================================================

def new_conversation():

    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.last_response_time = None
    st.session_state.conversation_id += 1


def clear_chat():

    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.last_response_time = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## NexaFlow")

    st.caption(
        "Internal AI Assistant"
    )

    st.divider()

    st.markdown(
        """
<div class="status-card">
<div class="status-label">Model</div>
<div class="status-value">Qwen2.5-1.5B · SFT</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="status-card">
<div class="status-label">Mode</div>
<div class="status-value">Policy-aware assistant</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="status-card">
<div class="status-label">Inference</div>
<div class="status-value">Local · 4-bit</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="status-card">
<div class="status-label">Context</div>
<div class="status-value">Independent SFT questions</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "### Conversation"
    )

    total_messages = len(
        st.session_state.messages
    )

    user_messages = sum(
        1
        for message in st.session_state.messages
        if message.get("role") == "user"
    )

    st.caption(
        f"Conversation #{st.session_state.conversation_id}"
    )

    st.caption(
        f"{total_messages} messages · "
        f"{user_messages} user questions"
    )

    if st.button(
        "＋ New conversation",
        use_container_width=True,
        key="new_conversation_button",
    ):

        new_conversation()
        st.rerun()

    if st.button(
        "Clear chat",
        use_container_width=True,
        key="clear_chat_button",
    ):

        clear_chat()
        st.rerun()

    st.divider()

    st.markdown(
        "### Can help with"
    )

    st.caption(
        "Company information"
    )

    st.caption(
        "Pricing and plans"
    )

    st.caption(
        "HR and annual leave"
    )

    st.caption(
        "Travel and expenses"
    )

    st.caption(
        "Security procedures"
    )

    st.caption(
        "Support and SLA"
    )

    st.caption(
        "Internal policies"
    )

    st.divider()

    st.caption(
        "NexaFlow AI · Experimental internal assistant"
    )

    st.caption(
        "Local inference"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="app-header">
<div class="app-title">NexaFlow AI Assistant</div>
<div class="app-subtitle">Company policy and internal knowledge assistant</div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
<div class="empty-state">
<div class="empty-title">How can I help?</div>
<div class="empty-subtitle">
Ask about documented NexaFlow company information,
policies, pricing, HR, security, travel, expenses,
support, or internal processes.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Who owns NexaFlow?",
            use_container_width=True,
            key="suggestion_owner",
        ):

            st.session_state.pending_prompt = (
                "Who owns NexaFlow?"
            )

            st.rerun()

        if st.button(
            "Explain the annual leave policy",
            use_container_width=True,
            key="suggestion_leave",
        ):

            st.session_state.pending_prompt = (
                "Explain the annual leave policy"
            )

            st.rerun()

    with col2:

        if st.button(
            "What is the Business plan price?",
            use_container_width=True,
            key="suggestion_price",
        ):

            st.session_state.pending_prompt = (
                "What is the Business plan price?"
            )

            st.rerun()

        if st.button(
            "I lost my MFA device. What should I do?",
            use_container_width=True,
            key="suggestion_mfa",
        ):

            st.session_state.pending_prompt = (
                "I lost my MFA device. What should I do?"
            )

            st.rerun()


# ============================================================
# DISPLAY HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(role):

        st.markdown(content)

        if (
            role == "assistant"
            and message.get("response_time") is not None
        ):

            st.markdown(
                (
                    '<div class="response-meta">'
                    f'{message["response_time"]:.2f}s · local inference'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_prompt = st.chat_input(
    "Ask NexaFlow..."
)


# ============================================================
# SUGGESTION INPUT
# ============================================================

if st.session_state.pending_prompt is not None:

    user_prompt = (
        st.session_state.pending_prompt
    )

    st.session_state.pending_prompt = None


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_prompt:

    user_prompt = user_prompt.strip()

    if user_prompt:

        # ----------------------------------------------------
        # STORE USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        # ----------------------------------------------------
        # SHOW USER
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                user_prompt
            )

        # ----------------------------------------------------
        # GENERATE ASSISTANT RESPONSE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            response_placeholder = (
                st.empty()
            )

            status_placeholder = (
                st.empty()
            )

            status_placeholder.markdown(
                """
<div class="muted-text">
Thinking...
</div>
""",
                unsafe_allow_html=True,
            )

            start_time = time.perf_counter()

            try:

                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    messages=st.session_state.messages,
                )

                if not response:

                    response = (
                        "I wasn't able to generate a response. "
                        "Please try rephrasing your NexaFlow question."
                    )

            except torch.cuda.OutOfMemoryError:

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                response = (
                    "The model ran out of GPU memory while processing "
                    "that request. Please try again."
                )

            except Exception as error:

                response = (
                    "I encountered an inference error while "
                    "processing that request."
                )

                st.error(
                    str(error)
                )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            status_placeholder.empty()

            response_placeholder.markdown(
                response
            )

            st.markdown(
                (
                    '<div class="response-meta">'
                    f"{elapsed:.2f}s · local inference"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # STORE ASSISTANT RESPONSE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "response_time": elapsed,
            }
        )

        st.session_state.last_response_time = elapsed

        st.rerun()