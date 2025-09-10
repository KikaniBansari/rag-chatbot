
import os
import json
import csv
import traceback
import pandas as pd

from dotenv import load_dotenv
from typing import List, Optional

# --- LangChain & Groq ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

# --- Existing Vector Store Code ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Configuration ---
VECTOR_STORE_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVER_K = 10
INPUT_FOLDER = "src"
GROQ_LLM_NAME = "llama-3.3-70b-versatile"

# --- Pipeline File Configuration (Corrected for your file) ---
INPUT_CSV_FILE = "C:/Users/bansa/Downloads/questions.csv"
INPUT_CSV_QUESTION_COLUMN = "Questions"  # <-- THIS LINE IS UPDATED
OUTPUT_JSON_FILE = "results.json"
OUTPUT_CSV_FILE = "results.csv"


# Load Groq API Key from .env file
load_dotenv()
# --- Structured Output Model using Pydantic ---
class GeneratedAnswer(BaseModel):
    """Defines the structured output for the LLM."""
    answer: str = Field(description="The concise, synthesized answer to the user's question, based only on the provided context.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating confidence that the answer is fully supported by the context.")
    supporting_sources: Optional[List[str]] = Field(description="A list of source URLs from the context that directly support the answer.")

def create_and_load_vector_store(input_folder_path: str) -> Chroma:
    """Creates or loads a Chroma vector store from JSON files."""
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

    all_documents = [
        Document(
            page_content=chunk,
            metadata={"source_url": result.get("cleaned_url", "Unknown URL")}
        )
        for json_file in json_files
        for item in json.load(open(os.path.join(input_folder_path, json_file), "r", encoding="utf-8"))
        for result in item.get("search_results", [])
        for chunk in result.get("content_chunks", [])
    ]
    if not all_documents:
        raise ValueError("No documents extracted from JSON files.")

    print(f"Embedding {len(all_documents)} document chunks...")
    return Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH,
    )

def find_relevant_context(vector_store: Chroma, query: str) -> List[Document]:
    """Searches the vector store for the top K most relevant document chunks."""
    search_results = vector_store.similarity_search(query=query, k=RETRIEVER_K)
    return search_results

def get_structured_answer_from_groq(question: str, context_chunks: List[Document]) -> Optional[GeneratedAnswer]:
    """Generates a structured answer using Groq and Llama3 based on provided context."""
    if not context_chunks:
        print("  [Warning] No context provided to generate an answer.")
        return None

    # --- Initialize the LLM ---
    llm = ChatGroq(temperature=0, model_name=GROQ_LLM_NAME)
    
    # --- Create the structured output chain ---
    structured_llm = llm.with_structured_output(GeneratedAnswer)

    # --- Define the prompt ---
    system_prompt = """You are an expert AI assistant. Your task is to analyze the provided context and answer the user's question based **only** on this information.
    - Synthesize a concise answer.
    - Provide a confidence score (0.0 to 1.0) on whether the context fully supports your answer.
    - List the source URLs from the context that directly support your answer.
    - If the context does not contain the answer, state that you cannot answer and set the confidence score to 0.0."""
    
    human_prompt = """**Context:**
    ---
    {context}
    ---
    **Question:** {question}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    # --- Format context and invoke the chain ---
    context_str = "\n\n".join(
        f"Source URL: {doc.metadata.get('source_url', 'N/A')}\nContent: {doc.page_content}"
        for doc in context_chunks
    )
    
    chain = prompt | structured_llm
    
    try:
        return chain.invoke({"question": question, "context": context_str})
    except Exception as e:
        print(f"  [Error] Failed to get answer from Groq: {e}")
        return None

def run_pipeline_from_file():
    """Main pipeline to read questions, generate structured answers, and save results."""
    try:
        print("--- Initializing RAG Pipeline with Groq ---")
        vector_store = create_and_load_vector_store(INPUT_FOLDER)
        
        if not os.path.exists(INPUT_CSV_FILE):
            raise FileNotFoundError(f"Input file not found: {INPUT_CSV_FILE}")

        questions_df = pd.read_csv(INPUT_CSV_FILE)
        all_results = []

        print(f"Processing {len(questions_df)} questions from {INPUT_CSV_FILE}...")
        
        for index, row in questions_df.iterrows():
            question = row[INPUT_CSV_QUESTION_COLUMN]
            if not isinstance(question, str) or not question.strip():
                print(f"  [Warning] Skipping row {index + 1} due to empty question.")
                continue

            print(f"\n[{index + 1}/{len(questions_df)}] Processing question: \"{question}\"")
            
            # 1. Retrieve Context
            context = find_relevant_context(vector_store, question)
            print(f"  > Retrieved {len(context)} context chunks.")
            
            # 2. Generate Structured Answer
            structured_answer = get_structured_answer_from_groq(question, context)
            
            # 3. Store result
            if structured_answer:
                result = {
                    "question": question,
                    "answer": structured_answer.answer,
                    "confidence_score": structured_answer.confidence_score,
                    "supporting_sources": structured_answer.supporting_sources or []
                }
                print(f"  > Generated Answer with Confidence: {structured_answer.confidence_score:.2f}")
            else:
                result = {
                    "question": question,
                    "answer": "Failed to generate an answer.",
                    "confidence_score": 0.0,
                    "supporting_sources": []
                }
            all_results.append(result)

        # --- 4. Save Outputs ---
        # Save to JSON
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4)
        
        # Save to CSV
        results_df = pd.DataFrame(all_results)
        # Convert list of sources to a string for CSV compatibility
        results_df['supporting_sources'] = results_df['supporting_sources'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else ''
        )
        results_df.to_csv(OUTPUT_CSV_FILE, index=False, quoting=csv.QUOTE_ALL)

        print("\n--- Pipeline Finished ---")
        print(f"Results saved to:")
        print(f"  - JSON: {OUTPUT_JSON_FILE}")
        print(f"  - CSV:  {OUTPUT_CSV_FILE}")

    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred during the pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline_from_file()
