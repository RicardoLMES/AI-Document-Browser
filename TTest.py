import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message


# Function to extract text from a list of PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text is not None:  # Check if text extraction is successful
                    text += page_text
                else:
                    st.error(f"Text could not be extracted from page {pdf_reader.pages.index(page) + 1} of {pdf.name}.")
        except Exception as e:
            st.error(f"Failed to read {pdf.name}: {e}")
    if not text:
        st.warning("No text could be extracted from the uploaded PDFs. Please check if they contain readable text.")
    return text


# Function to split a large text into manageable chunks for processing
def get_text_chunks(text):
    if not text:
        return []  # Return an empty list if there is no text to split
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create and save a vector store from text chunks for later retrieval
def load_vector_data(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local("faiss_vectorstore")
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vector data: {e}")
        return None


# Function to create a conversational chain that can handle queries using a retrieval model
def get_conversation_chain(query, chat_history=None):
    if chat_history is None:
        chat_history = []

    try:
        doc_search = FAISS.load_local(folder_path="faiss_vectorstore", embeddings=OpenAIEmbeddings())
        llm = ChatOpenAI()
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=doc_search.as_retriever(),
            chain_type="stuff"
        )
        response = conversation_chain({"question": query, "chat_history": chat_history})
        return response
    except Exception as e:
        st.error(f"Failed to create conversation chain: {e}")
        return {"answer": "Sorry, I couldn't process your query due to an error."}


# Function to initialize the environment and set Streamlit page configuration
def init():
    load_dotenv()
    # Test that the API key exists
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Please set the environment variable and restart the app.")
        st.stop()

    st.set_page_config(page_title="AI Document Reader Chat", page_icon=":books:")


# The main function where the Streamlit app logic resides
def main():
    # Initialize session state for conversation tracking
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    
    if "chat_response_history" not in st.session_state:
        st.session_state["chat_response_history"] = []

    # Set up the main page layout and input elements
    st.header("AI Document Reader Chat")  # Title of the page

    user_question = st.text_input("Ask a question about your document: ", key="user_question")
    chat_history = []

    # Process the user's question and get a response from the conversation chain
    if user_question:
        st.session_state["user_prompt_history"].append(user_question)
        with st.spinner("Thinking..."):
            response = get_conversation_chain(user_question, chat_history)
            st.session_state["chat_response_history"].append(response.get("answer", ""))

        chat_history.append((user_question, response.get("answer", "")))
    
    # Display the conversation history
    if chat_history:
        for prompt, response in zip(st.session_state["user_prompt_history"], st.session_state["chat_response_history"]):
            message(prompt, is_user=True)
            message(response)

    # Sidebar setup for uploading and processing PDF documents
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF documents here:", accept_multiple_files=True, type=['pdf'])
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)  # Turns PDF into text
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)  # Splits text into chunks
                            vectorstore = load_vector_data(text_chunks)  # Creates the vector store
                            if vectorstore is not None:
                                st.success("Documents processed successfully. You can now ask questions about your documents.")
                            else:
                                st.error("Failed to process documents for conversation.")
                    except Exception as e:
                        st.error(f"An error occurred while processing documents: {e}")
            else:
                st.warning("Please upload at least one PDF document.")


# Entry point for the script
if __name__ == '__main__':
    init()
    main()