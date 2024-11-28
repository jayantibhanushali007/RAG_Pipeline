import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from rich import print
import fitz
import re
import nltk
from nltk.tokenize import sent_tokenize



class RAGPipeline:
    def __init__(self, pdf_path):
         # Force CPU device
        self.device = "cpu"
        self.setup_nltk
        self.data = self.extract_text_from_pdf(pdf_path)
        self.embed_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        # Use smaller but more precise T5 model
        self.qa_pipeline = pipeline(
            "text2text-generation", 
            model="google/flan-t5-base",
            device=self.device
        )
        self.chunks = self.split_into_chunks(self.data)
        self.chunk_embeddings = self.create_embeddings()

    def clean_text(self, text):
        """
        Cleans the input text by performing the following operations:
        
        1. Removes URLs (http, https, www).
        2. Removes figure references (e.g., FIGURE 1.1).
        3. Removes chapter references (e.g., CHAPTER 1 OUTLINE).
        4. Removes special characters, keeping only alphanumeric characters, spaces, and basic punctuation (.,!?-).
        5. Removes multiple spaces and newlines, replacing them with a single space.
        
        Args:
            text (str): The input text to be cleaned.
        
        Returns:
            str: The cleaned text.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove figure references
        text = re.sub(r'FIGURE\s*\d+\.?\d*', '', text)
        # Remove chapter references
        text = re.sub(r'CHAPTER\s*\d+.*?OUTLINE', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def setup_nltk(self):
        """
        Sets up the NLTK library for use in the application.

        This method attempts to configure SSL to allow NLTK to download resources
        without verification. It then tries to download the 'punkt' tokenizer data
        from NLTK. If the download fails, it prints an error message and returns
        False, indicating that the setup was unsuccessful. If the download is
        successful, it returns True.

        Returns:
            bool: True if NLTK setup and download were successful, False otherwise.
        """
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        try:
            nltk.download('punkt_tab')
            pass
        except Exception as e:
            print(f"NLTK download failed: {e}")
            # Fallback to basic sentence splitting if NLTK fails
            return False
        return True
    
    def extract_text_from_pdf(self, pdf_input):
        """
        Extracts text from a PDF file or stream.
        This method handles both file path strings and file streams. It processes each page of the PDF,
        extracts the text, cleans it, and appends it to a list.
        Args:
            pdf_input (Union[str, bytes, file-like object]): The input PDF file path as a string, 
            or a file-like object containing the PDF data.
        Returns:
            List[str]: A list of cleaned text strings, one for each page of the PDF. 
            If an error occurs, an empty list is returned.
        Raises:
            Exception: If there is an error processing the PDF.
        """
        
        text = []
        try:
            # Handle both file path string and file stream
            if isinstance(pdf_input, str):
                pdf_document = fitz.open(pdf_input)
            else:
                # Convert to bytes if needed
                pdf_bytes = pdf_input.read() if hasattr(pdf_input, 'read') else pdf_input
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Process each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                cleaned_text = self.clean_text(page_text)
                if cleaned_text:
                    text.append(cleaned_text)
                    
            pdf_document.close()
            return text
        
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []
    
    def split_into_chunks(self, texts, chunk_size=500, overlap=100):
        """
        Splits a list of texts into smaller chunks of specified size with optional overlap.
        This method uses NLTK to tokenize sentences and then groups them into chunks.
        If the length of the current chunk plus the next sentence exceeds the chunk size,
        the current chunk is added to the list of chunks and a new chunk is started.
        Optionally, the last sentence of the current chunk can be included in the next chunk
        to create an overlap.
        Args:
            texts (list of str): List of texts to be split into chunks.
            chunk_size (int, optional): Maximum size of each chunk in characters. Default is 500.
            overlap (int, optional): Number of characters to overlap between chunks. Default is 100.
        Returns:
            list of str: List of text chunks.
        """
        
        if not self.setup_nltk():
            # Fallback chunking method
            print("Falling back to basic chunking method...")
            chunks = []
            for text in texts:
                words = text.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if chunk:
                        chunks.append(chunk)
            return chunks
        
        # Regular NLTK-based chunking
        chunks = []
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                if len(current_chunk) > 0 and current_length + len(sentence) > chunk_size:
                    chunks.append(' '.join(current_chunk))
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1]] if overlap > 0 else []
                    current_length = len(current_chunk[-1]) if current_chunk else 0
                
                current_chunk.append(sentence)
                current_length += len(sentence)
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks

    def create_embeddings(self):
        """
        Generates embeddings for the chunked data using the embed_model.

        This method encodes the data into embeddings using the embed_model and 
        ensures that the resulting embeddings are moved to the CPU.

        Returns:
            torch.Tensor: The embeddings of the data on the CPU.
        """
        embeddings = self.embed_model.encode(self.chunks, convert_to_tensor=True)
        return embeddings.cpu()  # Ensuring embeddings are on CPU

    def retrieve_context(self, query, top_k=3):
        """
        Retrieve the most relevant context chunks for a given query using Maximum Marginal Relevance (MMR).
        Args:
            query (str): The input query for which context needs to be retrieved.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 3.
        Returns:
            str: A string containing the concatenated top-k relevant context chunks.
        Notes:
            - The function first computes the cosine similarity between the query embedding and the chunk embeddings.
            - If there is a mismatch in the number of similarities and chunks, it falls back to a simple top-k selection.
            - The MMR algorithm is used to select diverse and relevant chunks iteratively.
            - The lambda parameter in MMR controls the trade-off between relevance and diversity.
        """
        # Get query embedding
        query_embedding = self.embed_model.encode([query], convert_to_tensor=True).cpu()
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)
        
        # Safety check for array sizes
        if len(similarities[0]) != len(self.chunks):
            print(f"Warning: similarity shape mismatch - similarities: {len(similarities[0])}, chunks: {len(self.chunks)}")
            # Fallback to simple top-k
            top_indices = np.argsort(similarities[0])[-min(top_k, len(self.chunks)):][::-1]
            return " ".join([self.chunks[i] for i in top_indices])
        
        # Initialize MMR
        selected_indices = []
        remaining_indices = np.arange(len(self.chunks))
        
        # Safe MMR selection
        while len(selected_indices) < top_k and len(remaining_indices) > 0:
            if not selected_indices:
                # First selection based on similarity
                idx = remaining_indices[np.argmax(similarities[0][remaining_indices])]
                selected_indices.append(idx)
                remaining_indices = remaining_indices[remaining_indices != idx]
            else:
                # MMR calculation
                lambda_param = 0.7
                remaining_sims = similarities[0][remaining_indices]
                diversity_sims = cosine_similarity(
                    self.chunk_embeddings[remaining_indices], 
                    self.chunk_embeddings[selected_indices]
                ).max(axis=1)
                
                mmr_scores = lambda_param * remaining_sims - (1 - lambda_param) * diversity_sims
                next_idx = remaining_indices[np.argmax(mmr_scores)]
                selected_indices.append(next_idx)
                remaining_indices = remaining_indices[remaining_indices != next_idx]
        
        return " ".join([self.chunks[i] for i in selected_indices])

    def generate_answer(self, query):
        """
        Generates an answer to the given query using a question-answering pipeline.

        This method retrieves the relevant context for the query and uses a 
        question-answering pipeline to generate a detailed and relevant answer 
        based on the context from a PDF file.

        Args:
            query (str): The question to be answered.

        Returns:
            str: The generated answer to the query.
        """
        context = self.retrieve_context(query)
        prompt = f"""You are a helpful assistant that specializes in answering Biology related questions ONLY.
        Based on the following context, provide a detailed and accurate answer to the question. 
        If the answer is not found within the context, answer "The question is out of context"  
        If the question has two parts, answer both parts.

        Context: {context}
        Question: {query}
        
        Answer:"""

        answer = self.qa_pipeline(
            prompt,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            num_beams=4,
            no_repeat_ngram_size=2
        )
        # Ensuring output is on CPU if it's a tensor
        if torch.is_tensor(answer):
            answer = answer.cpu()
        return answer[0]['generated_text']
    
#Chapter 1
	# What are the main characteristics of life?
	# How do scientists define a living organism?
	# What are the levels of organization in biological systems?
	# What is the difference between a prokaryotic and a eukaryotic cell?
	# What is the significance of the scientific method in biology?

#Chapter 2
    # What are the basic building blocks of matter?
	# What are atoms, and what are their components?
	# What is the difference between an element and a compound?
	# What is the importance of water in biological systems?
	# What are the properties of water that make it essential for life?


    # python3 -m nltk.downloader punkt