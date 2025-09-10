import os
import json
import traceback

from dotenv import load_dotenv
from typing import Optional, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Configuration ---
VECTOR_STORE_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVER_K = 10
SCORE_THRESHOLD = 0.5
INPUT_FOLDER = "src"

# Load environment variables if you have any (e.g., for API keys, though not used here)
load_dotenv()

def create_and_load_vector_store(input_folder_path: str) -> Chroma:
    """
    Creates a Chroma vector store from JSON files or loads an existing one.
    This version is explicitly configured to run on the CPU.
    """
    # Initialize embeddings model to run on CPU
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Loading existing vector store from {VECTOR_STORE_PATH}...")
        return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

    print(f"Creating new vector store from files in {input_folder_path}...")
    json_files = [f for f in os.listdir(input_folder_path) if f.endswith(".json")]
    if not json_files:
        raise ValueError(f"No JSON files found in {input_folder_path}")

    all_documents = []
    for json_file in json_files:
        file_path = os.path.join(input_folder_path, json_file)
        with open(file_path, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
        
        if not isinstance(processed_data, list):
            processed_data = [processed_data]

        documents_in_file = [
            Document(
                page_content=chunk,
                metadata={
                    "source_url": result.get("cleaned_url", "Unknown URL"),
                    "title": result.get("title", "No Title"),
                },
            )
            for item in processed_data
            for result in item.get("search_results", [])
            for chunk in result.get("content_chunks", [])
        ]
        all_documents.extend(documents_in_file)

    if not all_documents:
        raise ValueError("No documents were extracted from the JSON files.")

    print(f"Embedding {len(all_documents)} document chunks...")
    return Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH,
    )

def find_relevant_chunks(vector_store: Chroma, query: str) -> List[Document]:
    """
    Searches the vector store for relevant document chunks based on a score threshold.

    Args:
        vector_store: The Chroma vector store instance.
        query: The user's query string.

    Returns:
        A list of Document objects that meet the score threshold.
    """
    print("\n--- Searching for relevant chunks... ---")
    search_results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=RETRIEVER_K
    )

    if not search_results:
        print("  [Info] Retriever returned no documents.")
        return []

    # Filter results based on the score threshold
    relevant_docs = [doc for doc, score in search_results if score >= SCORE_THRESHOLD]
    
    print(f"  [Info] Found {len(relevant_docs)} chunks meeting the threshold ({SCORE_THRESHOLD}).")

    return relevant_docs

def main():
    """
    Main function to set up the vector store and handle user queries.
    """
    try:
        vector_store = create_and_load_vector_store(INPUT_FOLDER)

        print("\n--- Simple RAG Retriever Ready ---")
        print("Enter a query to find the most relevant document chunks.")

        while True:
            query = input("\nEnter your query (or 'exit'): ")
            if query.lower() == "exit":
                print("Exiting...")
                break
            if not query.strip():
                continue

            # Call the updated function to get a list of documents
            relevant_docs = find_relevant_chunks(vector_store, query)

            print("\n--- Result ---")
            if relevant_docs:
                print(f"Found {len(relevant_docs)} relevant piece(s) of information!")
                # Loop through and display each relevant document
                for i, doc in enumerate(relevant_docs):
                    print(f"\n--- Chunk {i+1} ---")
                    print("## Content")
                    print(doc.page_content)
                    print("\n## Source")
                    print(f"  - Title: {doc.metadata.get('title', 'N/A')}")
                    print(f"  - URL: {doc.metadata.get('source_url', 'N/A')}")
            else:
                print("I could not find any relevant information in the provided documents.")
            print("\n--------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()