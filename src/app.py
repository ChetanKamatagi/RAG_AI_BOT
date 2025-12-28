import streamlit as st
import time

st.set_page_config(
    page_title="Chetan's AI Portfolio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
    /* 🌑 DARK MODE COLORS */
    
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #444;
    }

    /* Text Colors (Force White/Light Grey) */
    h1, h2, h3, p, li, .stMarkdown, .stCaption {
        color: #E0E0E0 !important;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: #262730;
        border: 1px solid #444;
        border-radius: 15px;
    }
    
    /* User Message (Distinct Color) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E2a3a; /* Slight blue tint for user */
    }

    /* Input Box */
    .stTextInput input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #444;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #444;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        border-color: #00ADB5;
        color: #00ADB5;
    }
    
    /* Toast Styling */
    div[data-baseweb="toast"] {
        background-color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()


# A. Sidebar
with st.sidebar:
    st.markdown("### 👨‍💻 Chetan Kamatagi")
    st.caption("AI & Robotics Engineer")
    
    # Dark Mode Social Links
    st.markdown(
        """
        <div style="display: flex; gap: 15px; margin-bottom: 20px;">
            <a href="https://linkedin.com/in/chetan-kamatagi" target="_blank" style="color: #0077b5; text-decoration: none;">🔵 LinkedIn</a>
            <a href="https://github.com/ChetanKamatagi" target="_blank" style="color: #ffffff; text-decoration: none;">⚫ GitHub</a>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Callback to clear chat
    def reset_chat():
        st.session_state.messages = []

    # Mode Selector
    mode = st.radio(
        "Interact Mode:", 
        ["🤖 Chat with Chetan", "📄 Analyze Document"],
        on_change=reset_chat
    )
    
    # File Uploader Placeholder (Will be functional later)
    uploaded_file = None
    if mode == "📄 Analyze Document":
        st.divider()
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "md"])

# B. Main Header (Visible before DB loads)
if mode == "🤖 Chat with Chetan":
    st.title("👋 I'm Chetan's AI Agent")
    current_mode = "profile"
else:
    st.title("📄 Document Intelligence")
    current_mode = "document"

st.divider()

# --- 4. LAZY LOAD THE BACKEND (SHOW LOADING) ---

if "bot" not in st.session_state:
    with st.status("🚀 Booting up AI Brain...", expanded=True) as status:
        st.write("Loading Vector Database...")
        # Import inside the block to delay the heavy hit
        from chat import RAGSearch 
        st.session_state.bot = RAGSearch()
        st.write("Connecting to LLM...")
        time.sleep(0.5) # Fake tiny delay for UX smooth feel
        status.update(label="✅ System Ready!", state="complete", expanded=False)


if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. LOGIC & INTERACTION (The rest runs normally) ---

if uploaded_file and mode == "📄 Analyze Document":
    # Check if new file
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        with st.spinner("🧠 Indexing document..."):
            success, msg = st.session_state.bot.process_user_upload(uploaded_file)
            if success:
                st.toast("Document Ready!", icon="✅")
                st.session_state.last_file = uploaded_file.name
                # Add a system welcome message
                st.session_state.messages.append({"role": "assistant", "content": f"I've read **{uploaded_file.name}**. What would you like to know?"})
            else:
                st.error(msg)


for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if current_mode == "profile" and not st.session_state.messages:
    st.markdown("#### Suggested Queries:")
    col1, col2, col3 = st.columns(3)
    if col1.button("Tell me about your projects"):
        st.session_state.preset_query = "Tell me about your projects"
    if col2.button("What are your core skills?"):
        st.session_state.preset_query = "What are your core skills?"
    if col3.button("Download Resume Info"):
        st.session_state.preset_query = "Summarize your resume."


user_input = st.chat_input("Type your question here...")

if "preset_query" in st.session_state and st.session_state.preset_query:
    user_input = st.session_state.preset_query
    del st.session_state.preset_query

if user_input:
    # 1. Validation
    if current_mode == "document" and not uploaded_file:
        st.toast("⚠️ Upload a file first!", icon="🚨")
    else:
        # 2. User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # 3. AI Response
        with st.chat_message("assistant", avatar="🤖"):
            # Pass history for memory
            stream = st.session_state.bot.search_and_answer(
                user_input,
                mode=current_mode
            )
            response = st.write_stream(stream)
        
        # 4. Save to History
        st.session_state.messages.append({"role": "assistant", "content": response})