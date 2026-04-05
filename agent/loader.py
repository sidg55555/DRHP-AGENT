"""
loader.py - Loads all documents from the synthetic_data directory.

Walks the directory tree, reads every Markdown (.md) file, wraps its text
content in a dict, and tags each with source path and folder so downstream
nodes can reconstruct parent-child links.
"""

import re
from pathlib import Path


def _infer_doc_type(filename: str, content: str) -> str:
    """
    Heuristic document type inference from filename and content,
    so the classifier node has a warm start.
    """
    fname = filename.lower()
    if fname.startswith("sh7"):
        return "SH-7"
    if "board_meeting" in fname or "board meeting" in content.lower()[:200]:
        return "board_resolution"
    if fname.startswith("egm") and "notice" not in fname:
        return "egm_resolution"
    if "notice_of_egm" in fname or "notice of egm" in fname:
        return "notice_of_egm"
    if fname.startswith("moa"):
        return "moa_extract"
    return "unknown"


def load_all_documents(data_dir: str) -> list[dict]:
    """
    Recursively load all .md documents from data_dir.

    Returns a flat list of dicts with:
      - 'raw_text'     : full markdown content
      - 'filename'     : e.g. "sh7_002.md"
      - 'inferred_type': heuristic pre-classification
      - '_source_path' : relative path from data_dir
      - '_folder'      : parent folder name (e.g. "sh7_002")
    """
    documents = []
    data_path = Path(data_dir)

    for md_file in sorted(data_path.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            doc = {
                "filename": md_file.name,
                "raw_text": content,
                "inferred_type": _infer_doc_type(md_file.name, content),
                "_source_path": str(md_file.relative_to(data_path)),
                "_folder": str(md_file.parent.relative_to(data_path)),
            }
            documents.append(doc)
        except OSError as e:
            print(f"[WARN] Could not load {md_file}: {e}")

    print(f"[loader] Loaded {len(documents)} documents from {data_dir}")
    return documents
