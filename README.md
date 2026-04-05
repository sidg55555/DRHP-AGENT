# DRHP Capital Structure Drafting Agent

An AI-driven system that ingests Indian corporate filings (SH-7, board resolutions, EGM/AGM resolutions, MoA extracts) and produces a draft **Authorised Share Capital Change** table — the kind found in a DRHP filed with SEBI.



---

## Architecture

```
[load_documents] → [classify] → [link] → [extract] → [validate] → [table_builder] → [render]
```


---

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
echo "GOOGLE_API_KEY=sk-ant-..." > .env

# 3. Run the agent
python main.py
```

Outputs written to `output/`:
- `capital_structure.json` — structured table with flags
- `capital_structure.html` — rendered DRHP-style table

---

