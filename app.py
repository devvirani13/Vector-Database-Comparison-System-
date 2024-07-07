import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Annoy, FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import time
from langchain_community.document_loaders import CSVLoader

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def vector(pdf_docs, chunk_size, chunk_overlap):
    documents = []
    for csv_file in pdf_docs:
        print(f"Processing file: {csv_file.name}")
        with open(os.path.join("/tmp", csv_file.name), "wb") as f:
            f.write(csv_file.getbuffer())
        loader = CSVLoader(os.path.join("/tmp", csv_file.name))
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check and create Annoy index if not exists
    if not os.path.exists("my_annoy_index_and_docstore"):
        annoy_store = Annoy.from_documents(texts, embeddings)
        annoy_store.save_local("my_annoy_index_and_docstore")

    # Check and create FAISS index if not exists
    if not os.path.exists("faiss_index"):
        faiss_store = FAISS.from_documents(texts, embeddings)
        faiss_store.save_local("faiss_index")

    # Check and create Chroma index if not exists
    if not os.path.exists("./chroma_db"):
        chroma_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
        chroma_store.persist()

def get_conversational_chain(completion_model, temperature):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model=completion_model, temperature=temperature)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def ans_chroma(user_question, embeddings, top_k, completion_model, temperature):
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question, k=top_k)
    chain = get_conversational_chain(completion_model, temperature)
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response["output_text"])
    return response["output_text"]

def ans_faiss(user_question, embeddings, top_k, completion_model, temperature):
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=top_k)
    chain = get_conversational_chain(completion_model, temperature)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response["output_text"])
    return response["output_text"]

def ans_annoy(user_question, embeddings, top_k, completion_model, temperature):
    new_db = Annoy.load_local("my_annoy_index_and_docstore", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=top_k)
    chain = get_conversational_chain(completion_model, temperature)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response["output_text"])
    return response["output_text"]
import streamlit as st
import time

# Initialize session state variables
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = {}

def user_input(user_question, top_k, embedding_model, completion_model, temperature):
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    # Store question in session state
    st.session_state.questions.append(user_question)

    start_time = time.time()
    ans_c = ans_chroma(user_question, embeddings, top_k, completion_model, temperature)
    chroma_time = time.time() - start_time

    start_time = time.time()
    ans_f = ans_faiss(user_question, embeddings, top_k, completion_model, temperature)
    faiss_time = time.time() - start_time

    start_time = time.time()
    ans_a = ans_annoy(user_question, embeddings, top_k, completion_model, temperature)
    annoy_time = time.time() - start_time

    st.session_state.answers = {
        "chroma": ans_c,
        "faiss": ans_f,
        "annoy": ans_a,
        "chroma_time": chroma_time,
        "faiss_time": faiss_time,
        "annoy_time": annoy_time
    }

def display_results():
    if 'answers' in st.session_state:
        st.write("**Replies:**")
        st.write(f"**Chroma:** {st.session_state.answers['chroma']}")
        st.write(f"**FAISS:** {st.session_state.answers['faiss']}")
        st.write(f"**Annoy:** {st.session_state.answers['annoy']}")
        st.write(f"Chroma query time: {st.session_state.answers['chroma_time']:.2f} seconds")
        st.write(f"FAISS query time: {st.session_state.answers['faiss_time']:.2f} seconds")
        st.write(f"Annoy query time: {st.session_state.answers['annoy_time']:.2f} seconds")

def main():
    st.set_page_config(page_title="Vector Database Comparison System")
    st.header("Vector Database Comparison System 📊")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        
        chunk_size = st.slider("Chunk Size", min_value=1000, max_value=10000, value=5000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=5000, value=3000, step=100)
        
        embedding_model = st.text_input("Embedding Model", value="models/embedding-001")
        completion_model = st.text_input("Completion Model", value="gemini-pro")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5, step=1)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                vector(pdf_docs, chunk_size, chunk_overlap)
                st.success("Done")
    
    user_question = st.text_input("Ask a Question from the PDF Files", key="initial_question")

    if user_question:
        user_input(user_question, top_k, embedding_model, completion_model, temperature)
        display_results()

if __name__ == "__main__":
    main()



