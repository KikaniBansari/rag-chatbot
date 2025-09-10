import json
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import unquote, urlparse
import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, List, Dict, Any
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --- Pipeline state ---
class PipelineState(TypedDict):
    input_dir: str
    output_file: str
    file_paths: List[str]
    processed_data: List[Dict]
    text_splitter: RecursiveCharacterTextSplitter


# --- Utility: Clean Google Search URLs ---
def clean_google_url(url: str) -> str:
    if url.startswith("/url?q="):
        match = re.search(r'\?q=([^&]+)', url)
        if match:
            return unquote(match.group(1))
    return url



#  internet check function - doesn't scrape any data but gives out the json format with URL's
def is_internet_available(host="8.8.8.8", port=53, timeout=3) -> bool:
    """Check internet connectivity by trying to reach a known DNS server (Google DNS)."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


# --- Requests session ---
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})


# --- Scraper ---
def scrape_url_content(url: str) -> tuple[str, str]:
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname and parsed_url.hostname.endswith('.gov'):
            print(f"  -> Skipping government site: {url}")
            return url, ""
        
        response = session.get(url, timeout=20)
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "").lower()

        # --- PDF ---
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            try:
                reader = PdfReader(BytesIO(response.content))
                pdf_text = " ".join([page.extract_text() or "" for page in reader.pages])
                cleaned_text = re.sub(r"\s+", " ", pdf_text).strip()
                return url, cleaned_text
            except Exception:
                return url, ""

        # --- HTML ---
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'iframe']):
            tag.decompose()
        raw_text = ' '.join(soup.stripped_strings)
        cleaned_text = re.sub(r"\s+", " ", raw_text)
        cleaned_text = re.sub(r"(.)\1{5,}", r"\1", cleaned_text)  # collapse spammy repeats
        return url, cleaned_text.strip()

    except Exception as e:
        print(f"Failed {url}: {e}")
        return url, ""


# --- JSON processor ---
def process_json_file(input_filepath: str, text_splitter: RecursiveCharacterTextSplitter) -> Dict[str, Any]:
    print(f"\n--- Processing JSON: '{os.path.basename(input_filepath)}' ---")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'search_results' in data and isinstance(data['search_results'], list):
        urls_to_scrape = [clean_google_url(res['url']) for res in data['search_results'] if res.get('url')]
        for res, url in zip(data['search_results'], urls_to_scrape):
            res['cleaned_url'] = url

        internet = is_internet_available()
        if internet:
            print(" Internet detected → Scraping content...")
            scraped_content_map = {}
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(scrape_url_content, url): url for url in urls_to_scrape}
                for future in as_completed(future_to_url):
                    url, content = future.result()
                    if content:
                        scraped_content_map[url] = content
                        print(f"  -> Scraped {url} ({len(content)} chars)")
                    else:
                        print(f"  -> Skipped {url} (no usable text)")

            cleaned_results = []
            for result in data['search_results']:
                content = scraped_content_map.get(result.get('cleaned_url'))
                if content:
                    result['scraped_content'] = content
                    result['content_chunks'] = text_splitter.split_text(content)
                else:
                    result['scraped_content'] = ""
                    result['content_chunks'] = []
                cleaned_results.append(result)

            data['search_results'] = cleaned_results

        else:
            print(" No Internet detected → Skipping scraping, preserving format...")
            for result in data['search_results']:
                result['scraped_content'] = ""
                result['content_chunks'] = []

    return data


# --- Pipeline Nodes ---
def get_user_input_node(state: PipelineState) -> Dict[str, Any]:
    input_dir = input(" Enter input directory containing JSON files: ").strip()
    output_file = input(" Enter output file path (e.g., output.json): ").strip()
    return {"input_dir": input_dir, "output_file": output_file}


def find_files_node(state: PipelineState) -> Dict[str, Any]:
    file_paths = [os.path.join(state["input_dir"], f) for f in os.listdir(state["input_dir"]) if f.endswith(".json")]
    if not file_paths:
        raise FileNotFoundError(f"No JSON files found in {state['input_dir']}")
    return {"file_paths": file_paths}


def process_files_node(state: PipelineState) -> Dict[str, Any]:
    processed = []
    for filepath in state["file_paths"]:
        processed_data = process_json_file(filepath, state["text_splitter"])
        processed.append(processed_data)
    return {"processed_data": processed}


def save_output_node(state: PipelineState) -> None:
    with open(state["output_file"], "w", encoding="utf-8") as f:
        json.dump(state["processed_data"], f, ensure_ascii=False, indent=2)
    print(f"\n All processed data saved -> {state['output_file']}")


# --- Main execution block ---
if __name__ == "__main__":
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    current_state: PipelineState = {"text_splitter": splitter}

    try:
        current_state.update(get_user_input_node(current_state))
        current_state.update(find_files_node(current_state))
        current_state.update(process_files_node(current_state))
        save_output_node(current_state)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during the pipeline execution: {e}")
