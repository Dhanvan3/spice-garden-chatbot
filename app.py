import streamlit as st
from chatbot import load_knowledge_base, create_chatbot, ask

st.set_page_config(
    page_title="Spice Garden Assistant",
    page_icon="🍛",
    layout="centered"
)

# Header
st.image("https://img.icons8.com/emoji/96/curry-rice.png", width=80)
st.title("Spice Garden 🍛")
st.caption("Welcome! Ask me anything about our menu, hours, reservations and more.")
st.divider()

# Load chatbot once and cache it
@st.cache_resource
def get_chatbot():
    vectorstore = load_knowledge_base()
    return create_chatbot(vectorstore)

chatbot = get_chatbot()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new input
if prompt := st.chat_input("Ask me anything about Spice Garden..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and show bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(chatbot, prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})