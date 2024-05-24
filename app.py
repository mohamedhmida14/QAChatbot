from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

import numpy as np
import os
import streamlit as st

os.environ["HF_TOKEN"]= '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'
# Load the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

loader = PyPDFDirectoryLoader("./z_ClusterLab-QA/Documents")
documents = loader.load()

st.title("Chatgroq With Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        # Load HuggingFace BGE embeddings model
        embeddings_model = HuggingFaceBgeEmbeddings(model_name="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")

        # Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader("./Documents")
        st.session_state.docs = st.session_state.loader.load()
        
        # Document Splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        # Compute embeddings
        documents_texts = [doc.page_content for doc in st.session_state.final_documents]
        embeddings = embeddings_model.embed_documents(documents_texts)

        # Debug prints
        print(f"Number of documents: {len(st.session_state.final_documents)}")
        print(f"Embeddings shape: {embeddings.shape}")

        # Store embeddings in FAISS vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embeddings)

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
