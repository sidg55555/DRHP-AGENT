"""
main.py - Entry point for the DRHP Capital Structure Drafting Agent.

Usage:
    python main.py

Expects GOOGLE_API_KEY in environment or a .env file in the project root.
Reads synthetic_data/, writes output/ directory.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure agent package is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from agent.loader import load_all_documents
from agent.graph import build_graph


DATA_DIR = Path(__file__).parent / "synthetic_data"
OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- 1. Load all documents ---
    raw_documents = load_all_documents(str(DATA_DIR))
    if not raw_documents:
        print("[ERROR] No documents found in synthetic_data/. Exiting.")
        sys.exit(1)

    # --- 2. Build and invoke the LangGraph agent ---
    print("\n========== DRHP CAPITAL STRUCTURE AGENT ==========\n")
    graph = build_graph()

    initial_state = {
        "raw_documents": raw_documents,
        "classified_documents": [],
        "filing_groups": [],
        "extracted_data": [],
        "validated_data": [],
        "capital_structure_rows": [],
        "output_json": {},
        "output_html": "",
        "errors": [],
    }

    final_state = graph.invoke(initial_state)

    # --- 3. Write outputs ---
    json_path = OUTPUT_DIR / "capital_structure.json"
    html_path = OUTPUT_DIR / "capital_structure.html"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_state["output_json"], f, indent=2, ensure_ascii=False)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_state["output_html"])

    print(f"\n========== OUTPUT ==========")
    print(f"JSON → {json_path}")
    print(f"HTML → {html_path}")
    print(f"\nRows generated: {len(final_state['capital_structure_rows'])}")

    flags_total = sum(
        len(row.flags) for row in final_state["capital_structure_rows"]
    )
    print(f"Total flags raised: {flags_total}")

    if final_state.get("errors"):
        print(f"\n[ERRORS]: {final_state['errors']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
