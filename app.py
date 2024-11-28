import streamlit as st
from rag_pipeline import RAGPipeline
import os
import glob

def init_session_state():
    if 'rag' not in st.session_state:
        st.session_state.rag = None

def process_pdf_upload(uploaded_file):
    """Handle PDF upload and RAG pipeline initialization"""
    if st.session_state.rag is None:
        with st.spinner("Processing PDF..."):
            pdf_path = f"temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            
            st.session_state.rag = RAGPipeline(pdf_path)
            st.success("PDF processed successfully!")

def handle_question_answering():
    """Handle user questions and generate answers"""
    query = st.text_input("Enter your question:")
    
    if query and st.session_state.rag:
        with st.spinner("Fetching answer..."):
            answer = st.session_state['rag'].generate_answer(query)
            st.write(f"**Answer:** {answer}")

def cleanup_temp_files():
    """Clean up temporary PDF files"""
    temp_files = glob.glob("temp_*.pdf")
    for file in temp_files:
        os.remove(file)

def main():
    st.title("RAG Pipeline: Question Answering from PDF")
    
    # Initialize session state
    init_session_state()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        process_pdf_upload(uploaded_file)
        handle_question_answering()
    
    # Cleanup on app restart
    cleanup_temp_files()

if __name__ == "__main__":
    main()