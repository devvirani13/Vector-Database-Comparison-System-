# # Vector Database Comparison System 📊

- [About](#about)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Example Usage](#example-usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## About
This project allows you to upload PDF/CSV files, extract text, store them in three different vector databases (Chroma, FAISS, Annoy), and compare their performance in answering user questions.



## Features
- Upload Files: Upload PDF or CSV files containing text data.
- Vector Database Support: Supports three vector databases for storage and retrieval:
   - Chroma
   - FAISS
   - Annoy
- Question Answering: Perform question answering on uploaded documents using stored vectors.
P- erformance Comparison: Compare response times and answers from all three vector databases.

## Setup Instructions

### Prerequisites
- Python 3.6+
- Streamlit
- LangChain
- Google Generative AI API Key (set in .env file

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pdf-question-answering.git
   cd pdf-question-answering

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt

3. **Set up environment variables**:
   Create a '.env' file in the project root directory.
   Add your Google API key to the '.env' file:
   ```makefile
   GOOGLE_API_KEY=your-google-api-key



## Usage

1 Running the Application
   ```bash
      streamlit run main.py

2 Upload PDF/CSV Files:

- Upload one or more PDF or CSV files containing text data.
- Adjust chunk size and overlap as needed for text processing.

3 Choose Models:

- Enter the embedding model (default: models/embedding-001).
- Enter the completion model (default: gemini-pro).
- Adjust temperature and top K for question answering.

4 Submit & Process:

- Click on the "Submit & Process" button to start vectorization and indexing.
- Wait for processing to complete.

5 Ask a Question:

- Enter a question related to the uploaded documents.
- The system will display answers and query times from all three vector databases.

## Example Usage

- Upload a PDF file using the /upload endpoint.
- Use the /ask_question endpoint to ask questions about the uploaded PDF file.

## Dependencies

- Flask
- Flask-CORS
- langchain-community
- pdfminer
- PyPDF2
- pdfplumber
- pytesseract
- transformers
- torch
- easyocr
- ultralytics
- pdf2image
- dotenv

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [HuggingFace](https://huggingface.co)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [YOLO](https://github.com/ultralytics/ultralytics)
- [Flask](https://github.com/pallets/flask)


