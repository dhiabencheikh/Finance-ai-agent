import streamlit as st

st.set_page_config(page_title="Financial Analyst", layout="wide")
st.title("🤖 The Multi-Report Financial Analyst")

st.write("Welcome! Upload your financial documents on the left to get started.")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your 10-K reports, earnings calls, etc.",
                                        type=['pdf', 'txt'],
                                        accept_multiple_files=True)

st.subheader("Chat with your documents")
st.chat_input("Ask a question...")