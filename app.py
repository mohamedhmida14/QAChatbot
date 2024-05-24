import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Retrieve API keys from environment variables
hf_token = HF_TOKEN
groq_api_key = GROQ_API_KEY

# Streamlit App
st.title("Discover your future at ClusterLab")

# File uploader for the resume PDF
uploaded_file = st.file_uploader("Upload your resume (pdf) to see if you are a good fit !", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded PDF to the Documents folder
    documents_path = "./Documents"
    if not os.path.exists(documents_path):
        os.makedirs(documents_path)
    
    file_path = os.path.join(documents_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File {uploaded_file.name} uploaded successfully")

# Load PDF documents from directory
loader = PyPDFDirectoryLoader("./Documents")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(documents)

# HuggingFace Embeddings
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Create FAISS vector store
vectorstore = FAISS.from_documents(final_documents[:120], huggingface_embeddings)

# Set HuggingFace Hub API token
os.environ["HF_TOKEN"] = hf_token

# HuggingFace Hub LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    # Create RetrievalQA chain
    prompt_template = """
    Use the following piece of context to answer the question asked.
    Please try to provide the answer only based on the context
    
    {context}
    Question: {question}
    
    Helpful Answers:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Call the QA chain with our query
    result = retrievalQA.invoke({"query": query})

    # Display the result
    st.header("Result:")
    st.write(result['result'])

    # Display relevant documents
    with st.expander("Document Context"):
        for doc in result['source_documents']:
            st.write(doc.page_content)
            st.write("--------------------------------")
