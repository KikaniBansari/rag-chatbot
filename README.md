# Python Search Results Content Pipeline

This project processes JSON files containing search results, scrapes content from URLs (including HTML and PDF), cleans and chunks the text, and saves enriched results to a new JSON file. It is designed for research, data enrichment, and RAG (Retrieval-Augmented Generation) workflows.

## Features
- Scrapes and cleans content from web pages and PDFs
- Handles Google search result URLs
- Chunks text for downstream NLP tasks
- Parallel scraping for speed
- Skips government (.gov) sites

## Requirements
- Python 3.8+
- requests
- beautifulsoup4
- langchain
- PyPDF2

Install dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage
1. Place your JSON files (with a `search_results` list containing URLs) in a folder.
2. Run the main script:
   ```powershell
   python new_json.py
   ```
3. Enter the input folder and output file path when prompted.
4. The processed JSON will contain cleaned URLs, scraped content, and chunked text.

## Example JSON Input
```json
{
  "search_results": [
    {"url": "https://example.com"},
    {"url": "/url?q=https://another.com&sa=U"}
  ]
}
```

## Output
- The output JSON will have `cleaned_url`, `scraped_content`, and `content_chunks` for each result.

## Notes
- Large numbers of URLs are processed in parallel for speed.
- PDF content is extracted using PyPDF2.
- Text is chunked using LangChain's `RecursiveCharacterTextSplitter`.

## License
MIT
