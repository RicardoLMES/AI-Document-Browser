## AI Document Reader Chat

AI Document Reader Chat is a Streamlit-based web application that allows users to interact with their documents in a conversational manner. Utilizing state-of-the-art natural language processing and vector search technology, this app enables users to upload PDF documents, process them into searchable text chunks, and ask questions about the content of the documents.

## Features

- **PDF Text Extraction**: Upload PDF documents and extract their text for analysis.
- **Text Chunking**: Split large text extracts into manageable chunks for efficient processing.
- **Vector Data Storage**: Create and store vector representations of text chunks for quick retrieval.
- **Conversational Interface**: Engage with the document content through a chat interface, asking questions and receiving contextually relevant answers.
- **Session State Management**: Keep track of the conversation history within the session.

## Installation

Before you can run the app, you need to have Python installed on your system. The app has been tested with Python 3.8 and above.

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/ai-document-reader-chat.git
    cd ai-document-reader-chat
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your `.env` file with the necessary API keys:

    ```plaintext
    OPENAI_API_KEY='your_openai_api_key_here'
    ```

## Usage

To start the application, run the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

Once the app is running, navigate to http://localhost:8501 in your web browser to start using the AI Document Reader Chat.

## Usage How to Use
Upload Documents: Use the sidebar to upload one or more PDF documents.
Process Documents: Click the "Process" button to extract and chunk the text from the uploaded PDFs.
Ask Questions: Type your questions into the input field to receive answers based on the document content.
Review Conversation: Scroll through the conversation history to review past questions and answers.
Contributing
Contributions to the AI Document Reader Chat are welcome! Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Thanks to OpenAI for providing the GPT models and embeddings.
PyPDF2 for PDF text extraction capabilities.
Streamlit for creating an intuitive app interface.
Contact
For any queries or feedback, please open an issue in the GitHub repository or contact the maintainers directly.

Thank you for using AI Document Reader Chat!

## Although functional, there are a few bugs that still need resolving:
- Although not a bug, there is code and imports not being used, this is still to be fixed but I need to check exacly what I can and cannot delete.
- There is a weird loop happening where it will attempt to print the first query more than once, still investigating.
- When you first ask a question it gives you an answer alongside an error. Once you upload a document and then ask a question this does not happen, still investigating.

## What are the next steps to improve this app?
- FAISS Stores the informartion localy, perhaps a way to know what PDFs have been ingested already as to not get repeats.
- Add a way to catch errors when uploading the same file twice.
- Webscraping, a way to automatically ingest PDFs from the web.
- Wikipedia Ingestor, as well as PDFs I believe it could be of use to be able to add information from Wikipedia directly for an added layer of context. (Use case for this particular project would be to maintain an updated list of UK officials)


Helpful Documentation & Resources
https://docs.streamlit.io/library/get-started/main-concepts
https://python.langchain.com/docs/integrations/vectorstores/faiss
https://www.youtube.com/watch?v=cVA1RPsGQcw&ab_channel=PradipNichite