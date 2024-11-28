# RAG Pipeline: Question Answering from PDF

This project implements a Retrieval-Augmented Generation (RAG) pipeline for question answering from PDF documents. The pipeline extracts text from PDFs, splits the text into chunks, generates embeddings, retrieves relevant context, and generates answers to user queries.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Project Snapshots](#project-snapshots)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/rag-pipeline.git
    cd rag-pipeline
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Download NLTK data:

    ```sh
    python -m nltk.downloader punkt
    ```

## Usage

1. Run the Streamlit application:

    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a PDF file and enter your question to get an answer based on the content of the PDF.


## Architecture
![image](https://github.com/user-attachments/assets/9680affb-118b-4deb-b735-c166a686d9c6)


## Project Snapshots

![image](https://github.com/user-attachments/assets/0a655715-09fe-4cbd-a1cb-d1d1be775334)

![image](https://github.com/user-attachments/assets/23379a03-27e4-4467-8c34-042aea63170a)

![image](https://github.com/user-attachments/assets/982ca618-2a61-4ccf-b182-8e7bc6b8db19)


