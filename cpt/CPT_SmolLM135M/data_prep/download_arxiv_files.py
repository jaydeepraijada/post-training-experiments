import re
import concurrent.futures

import arxiv
import requests
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader


output_dir = Path("papers")
output_dir.mkdir(exist_ok=True)


def query_arxiv(query: str = "transformer large language model", max_results: int = 50) -> list[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in search.results():
        arxiv_id = re.sub(r"v\d+$", "", result.entry_id.split("/")[-1])
        papers.append({
            "arxiv_id": arxiv_id,
            "title": " ".join(result.title.split()),
            "published": result.published,
        })
    return papers


def download_paper(arxiv_id: str) -> str:
    url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = PdfReader(BytesIO(resp.content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def process_paper(arxiv_id: str, i: int, total: int, title: str | None = None):
    filepath = output_dir / f"{arxiv_id}.txt"
    title_str = f"  {title[:72]}" if title else ""
    print(f"[{i}/{total}] {arxiv_id}{title_str}")
    if filepath.exists():
        print(f"         -> exists, skipping\n")
        return
    try:
        text = download_paper(arxiv_id)
        filepath.write_text(text, encoding="utf-8")
        print(f"         -> {filepath}  ({len(text):,} chars)\n")
    except Exception as e:
        print(f"         -> failed: {e}\n")


def auto_main():
    query = 'submittedDate:[20240101 TO 20260201]'
    print(f"Querying arxiv: {query}\n")
    papers = query_arxiv(query=query, max_results=200)
    print(f"Downloading {len(papers)} papers\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_paper, p["arxiv_id"], i, len(papers), p["title"])
            for i, p in enumerate(papers, 1)
        ]
        concurrent.futures.wait(futures)


def main():
    arxiv_ids = [line.strip() for line in open("arxiv_ids.txt") if line.strip()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_paper, arxiv_id, i, len(arxiv_ids))
            for i, arxiv_id in enumerate(arxiv_ids, 1)
        ]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    auto_main()
