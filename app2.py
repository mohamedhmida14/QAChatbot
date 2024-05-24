import streamlit as st
import os
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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
os.environ["HF_TOKEN"]= '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'
groq_api_key = os.getenv('GROQ_API_KEY')
# HuggingFace Hub LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Streamlit App
st.title("HuggingFace LLM and FAISS VectorStore Demo")

query = st.text_area("Enter your question:")

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
        llm=hf,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Call the QA chain with our query
    result = retrievalQA.invoke({"query": query})

    # Display the result
    st.header("Result:")
    st.write(result['answer'])

    # Display relevant documents
    with st.expander("Document Context"):
        for doc in result['context']:
            st.write(doc.page_content)
            st.write("--------------------------------")
