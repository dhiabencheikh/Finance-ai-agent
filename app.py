import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time # To simulate processing

# --- 1. SET UP THE ENVIRONMENT ---

# Load the secret API key from the .env file
load_dotenv()
try:
    Open_AI_Key = os.environ['Open_AI_Key']
except KeyError:
    st.error("Open_AI_Key not found. Please set it in your .env file.")
    st.stop()

# --- 2. DEFINE CORE LOGIC FUNCTIONS ---

def get_vectorstore_from_files(uploaded_files):
    """
    Takes a list of uploaded files, processes them, and returns a FAISS vector store.
    This function will be cached by Streamlit to avoid re-processing.
    """
    if not uploaded_files:
        return None

    # This is where we'll store the text from all PDFs
    all_pages_text = ""
    for pdf in uploaded_files:
        # We need to save the uploaded file temporarily to disk so PyPDFLoader can read it
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        
        loader = PyPDFLoader(pdf.name)
        pages = loader.load_and_split()
        for page in pages:
            all_pages_text += page.page_content + "\n\n"
        
        # Clean up the temporary file
        os.remove(pdf.name)

    # Split the combined text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text(all_pages_text)

    # Initialize the free, open-source embedding model
    # This will download the model (once) to our Codespace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the vector store from the text chunks
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    
    return vector_store

def get_rag_chain(_vector_store):
    """
    Takes a vector store and returns a complete LangChain RAG chain
    ready to answer questions.
    """

    # 1. Initialize the LLM (using OpenAI)
    llm = ChatOpenAI(openai_api_key=Open_AI_Key, model_name="gpt-3.5-turbo")

    # 2. Create the Retriever
    retriever = _vector_store.as_retriever()

    # 3. Define the Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert financial analyst assistant. 
        Answer the user's question based *only* on the following context:

        <context>
        {context}
        </context>

        Question: {input}

        Answer:
        """
    )

    # 4. Create the "Stuff" Chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 5. Create the final Retrieval Chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# --- 3. BUILD THE STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Financial Analyst", layout="wide")
st.title("🤖 The Multi-Report Financial Analyst")

# Initialize session state variables
# This is a critical part of Streamlit. It helps "remember" things.
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed" not in st.session_state:
    st.session_state.processed = False

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your 10-K reports, earnings calls, etc.",
                                        type=['pdf'],  # Only allow PDFs
                                        accept_multiple_files=True)

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing documents... This may take a moment."):
                # 1. Create the vector store
                vector_store = get_vectorstore_from_files(uploaded_files)
                
                # 2. Create the RAG chain
                st.session_state.conversation_chain = get_rag_chain(vector_store)
                
                # 3. Mark as processed
                st.session_state.processed = True
                
            st.success("Documents processed! You can now ask questions.")

# Main chat interface
st.subheader("Chat with your documents")

# Display a message if documents haven't been processed
if not st.session_state.processed:
    st.info("Please upload and process your documents in the sidebar to start chatting.")

# Display past chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The chat input box
# We disable it until the documents are processed
if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.processed):
    
    # 1. Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get the AI's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # This is where we call the RAG chain!
            response = st.session_state.conversation_chain.invoke({"input": prompt})
            
            # The actual answer is nested in the 'answer' key
            answer = response["answer"]
            
            st.markdown(answer)
    
    # 3. Add AI response to history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
