import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from llm2vec import LLM2Vec
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]= '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'
# Load the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Chatgroq With Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        # Load LLM2Vec model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
        config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True)
        model = AutoModel.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True, config=config, torch_dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu")
        model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")

        l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

        # Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader("./Documents")
        st.session_state.docs = st.session_state.loader.load()
        
        # Document Splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        # Compute embeddings using LLM2Vec model
        documents_texts = [doc.page_content for doc in st.session_state.final_documents]
        embeddings = l2v.encode(documents_texts)

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
