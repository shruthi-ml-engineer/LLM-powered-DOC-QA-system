import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# 📥 Load environment variables (ensure .env file has OPENAI_API_KEY=your_key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Initialize LLM and Embeddings
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

st.title("📄 Chat with your PDF using GPT")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Step 1: Load PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Step 2: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(pages)

    # Step 3: Create Vector Store
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Step 4: Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Step 5: Ask questions
    st.success("PDF Processed! You can now ask questions.")
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(query)
            st.write("### Answer:")
            st.write(result)
