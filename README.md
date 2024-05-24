# Personalized QA Chatbot with RAG using Streamlit

This project is a personalized Question Answering (QA) chatbot built using the Retrieve-Augment-Generate (RAG) model, integrated with Streamlit for the user interface. The chatbot allows users to ask questions and receive personalized answers based on uploaded resumes (PDF).

## Technologies Used

- **Streamlit**: Streamlit is used for building the web-based user interface. It provides an easy-to-use interface for users to interact with the chatbot.
- **Langchain Library**: Langchain is a library used for various natural language processing (NLP) tasks, including document loading, text splitting, embeddings, and prompts. It simplifies the process of integrating NLP models into the application.
- **Hugging Face Transformers**: Hugging Face Transformers library is utilized for leveraging pre-trained language models. In this project, the BGE (Big Generalized Embedding) model is used for generating embeddings and answering questions.
- **FAISS**: FAISS (Facebook AI Similarity Search) is used for efficient similarity search and nearest neighbor search. It allows for fast retrieval of relevant documents based on user queries.
- **ChatGroq**: ChatGroq is a library used for integrating with Hugging Face Hub, allowing access to various language models for conversational AI tasks.

## Idea and Functionality

The main idea behind this project is to create a personalized QA chatbot that can provide tailored answers based on the user's uploaded resume. The chatbot leverages the RAG model, which combines retrieval-based and generation-based approaches to generate context-aware responses.

### How it Works

1. **Upload Resume**: Users upload their resumes (in PDF format) through the web interface.
2. **Processing**: The uploaded resume is processed to extract relevant information.
3. **Question Input**: Users can input questions related to their career, skills, experiences, etc.
4. **Answer Generation**: The chatbot retrieves relevant information from the uploaded resume and generates personalized answers using the RAG model.
5. **Display Results**: The chatbot displays the generated answers along with relevant document context for transparency.
