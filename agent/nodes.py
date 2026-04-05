"""
nodes.py - All LangGraph node functions for the DRHP Capital Structure Agent.

Documents are plain markdown text dicts with keys:
  filename, raw_text, inferred_type, _source_path, _folder

Every node works with this structure end-to-end.
Each function: (state: AgentState) -> dict  (returns only keys it updates)
"""

import json
import re
import time
import dataclasses
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState, CapitalStructureRow, CapitalSnapshot

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _invoke(system: str, user: str) -> str:
    prompt = [SystemMessage(content=system), HumanMessage(content=user)]
    wait = 10
    while True:
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                print(f"  [Rate limited] Sleeping {wait}s before retry...")
                time.sleep(wait)
                wait = min(wait * 2, 60)  # 10s → 20s → 40s → 60s cap
            else:
                raise e


def _parse_json(text: str) -> dict:
    """Strip markdown code fences then parse JSON."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    return json.loads(cleaned)


def _format_group_for_llm(group: dict) -> str:
    """
    Render a filing group as readable plain text for LLM prompts.
    Passes raw_text directly — avoids JSON-serialising it (broken escaping + huge size).
    """
    sh7 = group["sh7"]
    parts = [
        f"=== SH-7 FILING: {sh7.get('filename')} ===\n{sh7.get('raw_text', '')[:3000]}"
    ]
    for att in group.get("attachments", []):
        att_type = att.get("_classification", {}).get("confirmed_type", att.get("inferred_type", "unknown"))
        parts.append(
            f"=== ATTACHMENT [{att_type.upper()}]: {att.get('filename')} ===\n"
            f"{att.get('raw_text', '')[:2000]}"
        )
    return "\n\n".join(parts)


def _extract_filing_date(sh7_text: str) -> str:
    """
    Pull the eForm filing date from SH-7 markdown for chronological sorting.
    Falls back through several patterns; returns '9999-99-99' if nothing found.
    """
    patterns = [
        r"eForm filing date\s+(\d{2}/\d{2}/\d{4})",
        r"Date of signing\s+(\d{2}/\d{2}/\d{4})",
        r"held on\s*\|?\s*(\d{2}/\d{2}/\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, sh7_text)
        if m:
            d, mo, y = m.group(1).split("/")
            return f"{y}-{mo}-{d}"
    return "9999-99-99"


# ---------------------------------------------------------------------------
# Node 1: classify_node
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """
You are a document classification expert for Indian corporate filings under the Companies Act, 2013.

Given a document filename and its markdown text, identify its type. Choose EXACTLY one label:

  SH-7             — Form SH-7 filed with MCA to notify the Registrar of a share capital alteration
  PAS-3            — Form PAS-3 return of allotment of shares
  board_resolution — Certified true copy of a Board of Directors meeting resolution
  egm_resolution   — Certified true copy of an Extra Ordinary General Meeting resolution
  agm_resolution   — Certified true copy of an Annual General Meeting resolution
  notice_of_egm    — Formal notice convening an Extra Ordinary General Meeting
  moa_extract      — Memorandum of Association (original or altered) containing the capital clause
  unknown          — Cannot be determined

Return ONLY a raw JSON object (no markdown fences, no extra text):
{
  "filename": "<filename>",
  "confirmed_type": "<one label from the list above>",
  "confidence": "high|medium|low",
  "reason": "<one sentence>"
}
""".strip()


def classify_node(state: AgentState) -> dict:
    """Classify each raw markdown document by type using the LLM."""
    print("[classify_node] Classifying documents...")
    classified = []

    for doc in state["raw_documents"]:
        prompt = (
            f"Filename: {doc.get('filename')}\n"
            f"Filename-based type hint: {doc.get('inferred_type')}\n\n"
            f"Document content (first 2500 chars):\n"
            f"{doc.get('raw_text', '')[:2500]}"
        )
        raw = _invoke(CLASSIFY_SYSTEM, prompt)

        try:
            result = _parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            result = {
                "filename": doc.get("filename"),
                "confirmed_type": doc.get("inferred_type", "unknown"),
                "confidence": "low",
                "reason": "LLM parse error — fell back to filename heuristic",
            }

        enriched = {**doc, "_classification": result}
        classified.append(enriched)
        print(f"  → {doc.get('filename', '?'):<42} "
              f"{result.get('confirmed_type')} ({result.get('confidence')})")

    return {"classified_documents": classified}


# ---------------------------------------------------------------------------
# Node 2: link_node
# ---------------------------------------------------------------------------

def link_node(state: AgentState) -> dict:
    """
    Group documents into filing packages: one SH-7 + its attachments.
    All documents in the same _folder belong to the same filing.
    """
    print("[link_node] Linking documents to parent SH-7 filings...")

    folder_groups: dict[str, dict] = {}

    for doc in state["classified_documents"]:
        folder = doc.get("_folder", "unknown")
        if folder not in folder_groups:
            folder_groups[folder] = {"sh7": None, "attachments": []}

        confirmed_type = doc.get("_classification", {}).get("confirmed_type", "unknown")
        if confirmed_type == "SH-7":
            folder_groups[folder]["sh7"] = doc
        else:
            folder_groups[folder]["attachments"].append(doc)

    filing_groups = []
    for folder, group in sorted(folder_groups.items()):
        if group["sh7"] is None:
            print(f"  [WARN] No SH-7 found in folder '{folder}' — skipping")
            continue
        filing_groups.append(group)
        att_names = [a.get("filename", "?") for a in group["attachments"]]
        print(f"  → {folder}: SH-7={group['sh7'].get('filename')} | attachments={att_names}")

    return {"filing_groups": filing_groups}


# ---------------------------------------------------------------------------
# Node 3: extract_node
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """
You are a financial data extraction specialist for Indian corporate filings.

You will receive the full text of an SH-7 filing and its attachment documents.
Extract the authorised share capital change event as structured data.

Rules:
- The SH-7 form is the primary source of truth for capital figures and dates.
- Attachments (board resolution, EGM/AGM resolution, MoA) corroborate the SH-7.
- If a value is clearly stated in at least one document, extract it.
- If a value is ABSENT from ALL documents, set it to null — never guess.
- Note discrepancies between documents in raw_flags (empty list if none).
- For sh7_filename, use the exact filename shown in the SH-7 header.
- For incorporation events (no prior capital), set capital_before fields to null.
- For incorporation events, set date_of_shareholder_meeting and meeting_type to null.
- All monetary values are in INR.

Return ONLY a raw JSON object (no markdown fences):
{
  "sh7_filename": "<exact SH-7 filename, e.g. sh7_002.md>",
  "event": "<e.g. 'Incorporation' or 'Increase in Authorised Capital'>",
  "date_of_shareholder_meeting": "<YYYY-MM-DD or null>",
  "meeting_type": "<AGM|EGM|null>",
  "capital_before": {
    "equity_shares": <int or null>,
    "equity_face_value": <float or null>,
    "equity_total": <float or null>,
    "preference_shares": <int or null>,
    "preference_face_value": <float or null>,
    "preference_total": <float or null>,
    "total_authorised": <float or null>
  },
  "capital_after": {
    "equity_shares": <int or null>,
    "equity_face_value": <float or null>,
    "equity_total": <float or null>,
    "preference_shares": <int or null>,
    "preference_face_value": <float or null>,
    "preference_total": <float or null>,
    "total_authorised": <float or null>
  },
  "supporting_filenames": ["<list of attachment filenames that were present>"],
  "raw_flags": ["<specific anomaly — empty list if documents are consistent>"]
}
""".strip()


def extract_node(state: AgentState) -> dict:
    """Extract structured capital data from each filing group's markdown text."""
    print("[extract_node] Extracting capital data...")
    extracted = []

    for group in state["filing_groups"]:
        group_text = _format_group_for_llm(group)
        raw = _invoke(EXTRACT_SYSTEM,
                      f"Extract capital structure data from these documents:\n\n{group_text}")

        try:
            result = _parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            result = {
                "sh7_filename": group["sh7"].get("filename", "unknown"),
                "event": "EXTRACTION_ERROR",
                "date_of_shareholder_meeting": None,
                "meeting_type": None,
                "capital_before": None,
                "capital_after": None,
                "supporting_filenames": [],
                "raw_flags": ["EXTRACTION_PARSE_ERROR: LLM returned unparseable output"],
            }

        extracted.append(result)
        flags = result.get("raw_flags", [])
        print(f"  → {result.get('sh7_filename')} | flags: {flags if flags else 'none'}")

    return {"extracted_data": extracted}


# ---------------------------------------------------------------------------
# Node 4: validate_node
# ---------------------------------------------------------------------------

VALIDATE_SYSTEM = """
You are a compliance validator for DRHP capital structure filings.

You receive:
1. Extracted capital data for one SH-7 event (JSON).
2. The original document texts to cross-reference.

Validate the extraction and raise structured flags for any issues.

Flag codes — use EXACTLY these:
  MISSING_EGM_RESOLUTION   — No EGM or AGM resolution document is in the filing package
  MISSING_BOARD_RESOLUTION — No board resolution document is present
  MISSING_MOA_EXTRACT      — No Memorandum of Association extract is present
  MOA_CAPITAL_MISMATCH     — MoA stated total ≠ arithmetic sum of its share classes
  CROSS_DOC_DATE_CONFLICT  — A date in one document conflicts with another document
  UNCONFIRMED_FIELD        — A value in the SH-7 is not corroborated by any attachment
  CAPITAL_MATH_ERROR       — Arithmetic is internally wrong within a single document
  NO_SHAREHOLDER_MEETING   — Incorporation event; no EGM/AGM applicable (informational only)

Rules:
- Check for all three attachment types: board resolution, EGM/AGM resolution, MoA.
- For incorporation: NO_SHAREHOLDER_MEETING is expected. Do not also raise MISSING_EGM_RESOLUTION.
- is_clean = true ONLY when flags list is empty.
- Be specific: include filename + exact discrepancy in each flag description.
- For validated_capital fields, use the most reliable values (SH-7 takes precedence over attachments).

Return ONLY a raw JSON object (no markdown fences):
{
  "sh7_filename": "<filename>",
  "is_clean": <true|false>,
  "flags": ["<FLAG_CODE: specific explanation>"],
  "validated_capital_before": {
    "equity_shares": <int or null>,
    "equity_face_value": <float or null>,
    "equity_total": <float or null>,
    "preference_shares": <int or null>,
    "preference_face_value": <float or null>,
    "preference_total": <float or null>,
    "total_authorised": <float or null>
  },
  "validated_capital_after": {
    "equity_shares": <int or null>,
    "equity_face_value": <float or null>,
    "equity_total": <float or null>,
    "preference_shares": <int or null>,
    "preference_face_value": <float or null>,
    "preference_total": <float or null>,
    "total_authorised": <float or null>
  },
  "date_of_shareholder_meeting": "<YYYY-MM-DD or null>",
  "meeting_type": "<AGM|EGM|null>",
  "event": "<string>",
  "supporting_filenames": ["<filenames>"]
}
""".strip()


def validate_node(state: AgentState) -> dict:
    """Validate extractions against original markdown docs; produce structured flags."""
    print("[validate_node] Validating extractions...")
    validated = []

    for extraction in state["extracted_data"]:
        sh7_filename = extraction.get("sh7_filename", "")

        # Find matching filing group by SH-7 filename
        group = next(
            (g for g in state["filing_groups"]
             if g["sh7"].get("filename") == sh7_filename),
            None,
        )

        extraction_str = json.dumps(extraction, indent=2)
        docs_text = _format_group_for_llm(group) if group else "(original documents unavailable)"

        payload = (
            f"EXTRACTED DATA:\n{extraction_str}\n\n"
            f"ORIGINAL DOCUMENTS FOR CROSS-REFERENCE:\n{docs_text}"
        )

        raw = _invoke(VALIDATE_SYSTEM, f"Validate this extraction:\n\n{payload}")

        try:
            result = _parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            result = {
                "sh7_filename": sh7_filename,
                "is_clean": False,
                "flags": ["VALIDATION_PARSE_ERROR: LLM returned unparseable output"],
                "validated_capital_before": extraction.get("capital_before"),
                "validated_capital_after": extraction.get("capital_after"),
                "date_of_shareholder_meeting": extraction.get("date_of_shareholder_meeting"),
                "meeting_type": extraction.get("meeting_type"),
                "event": extraction.get("event", ""),
                "supporting_filenames": extraction.get("supporting_filenames", []),
            }

        validated.append(result)
        if result.get("is_clean"):
            print(f"  → {result.get('sh7_filename')} | ✓ CLEAN")
        else:
            print(f"  → {result.get('sh7_filename')} | ⚠  {result.get('flags', [])}")

    return {"validated_data": validated}


# ---------------------------------------------------------------------------
# Node 5: table_builder_node
# ---------------------------------------------------------------------------

def _to_snapshot(d: dict | None) -> CapitalSnapshot | None:
    if not d:
        return None
    return CapitalSnapshot(
        equity_shares=d.get("equity_shares"),
        equity_face_value=d.get("equity_face_value"),
        equity_total=d.get("equity_total"),
        preference_shares=d.get("preference_shares"),
        preference_face_value=d.get("preference_face_value"),
        preference_total=d.get("preference_total"),
        total_authorised=d.get("total_authorised"),
    )


def table_builder_node(state: AgentState) -> dict:
    """Assemble validated data into a chronological capital structure table."""
    print("[table_builder_node] Building capital structure table...")

    # Build filing-date lookup from SH-7 raw_text via regex
    filing_date_map: dict[str, str] = {}
    for doc in state["raw_documents"]:
        fname = doc.get("filename", "")
        if fname.startswith("sh7"):
            filing_date_map[fname] = _extract_filing_date(doc.get("raw_text", ""))

    sorted_validated = sorted(
        state["validated_data"],
        key=lambda v: filing_date_map.get(v.get("sh7_filename", ""), "9999-99-99"),
    )

    rows = []
    for v in sorted_validated:
        row = CapitalStructureRow(
            date_of_shareholder_meeting=v.get("date_of_shareholder_meeting"),
            meeting_type=v.get("meeting_type"),
            capital_before=_to_snapshot(v.get("validated_capital_before")),
            capital_after=_to_snapshot(v.get("validated_capital_after")) or CapitalSnapshot(),
            source_document=v.get("sh7_filename", "UNKNOWN"),
            supporting_docs=v.get("supporting_filenames", []),
            flags=v.get("flags", []),
            event=v.get("event", ""),
        )
        rows.append(row)
        flag_str = row.flags if row.flags else "none"
        print(f"  → {row.source_document} | {row.event} | flags: {flag_str}")

    return {"capital_structure_rows": rows}


# ---------------------------------------------------------------------------
# Node 6: render_node
# ---------------------------------------------------------------------------

def _snapshot_to_str(snap: CapitalSnapshot | None, flagged: bool = False) -> str:
    """Format a CapitalSnapshot as a DRHP-style human-readable string."""
    if snap is None:
        return "—"

    has_eq   = bool(snap.equity_shares and snap.equity_face_value)
    has_pref = bool(snap.preference_shares and snap.preference_face_value)
    total    = snap.total_authorised

    if not has_eq and not has_pref:
        return "⚑ UNCONFIRMED" if flagged else "—"

    if has_eq and not has_pref:
        return (
            f"₹ {total:,.0f} divided into "
            f"{snap.equity_shares:,} Equity Shares of ₹ {snap.equity_face_value:.0f} each"
        )

    parts = []
    if has_eq:
        parts.append(f"{snap.equity_shares:,} Equity Shares of ₹ {snap.equity_face_value:.0f} each")
    if has_pref:
        parts.append(f"{snap.preference_shares:,} Preference Shares of ₹ {snap.preference_face_value:.0f} each")

    return f"₹ {total:,.0f} divided into " + " and ".join(parts)


def render_node(state: AgentState) -> dict:
    """Render the capital structure table as JSON and HTML."""
    print("[render_node] Rendering outputs...")
    rows = state["capital_structure_rows"]

    # ── JSON ──
    json_rows = []
    for row in rows:
        json_rows.append({
            "date_of_shareholder_meeting": row.date_of_shareholder_meeting or "On incorporation",
            "meeting_type": row.meeting_type or "—",
            "event": row.event,
            "capital_before": dataclasses.asdict(row.capital_before) if row.capital_before else None,
            "capital_after":  dataclasses.asdict(row.capital_after),
            "source_document": row.source_document,
            "supporting_docs": row.supporting_docs,
            "flags":    row.flags,
            "is_clean": len(row.flags) == 0,
        })

    output_json = {
        "company": "TechNova Private Limited",
        "cin":     "U72900MH2015PTC267801",
        "table_title":  "Authorised Share Capital Change History",
        "generated_by": "DRHP Capital Structure Agent (LangGraph + Claude)",
        "rows": json_rows,
    }

    # ── HTML ──
    html_rows = ""
    for row in rows:
        if row.flags:
            items     = "".join(f"<li>{f}</li>" for f in row.flags)
            flag_cell = f'<ul class="flags">{items}</ul>'
            row_class = "flagged"
        else:
            flag_cell = '<span class="ok">✓ Clean</span>'
            row_class = "clean"

        before_str   = _snapshot_to_str(row.capital_before)
        after_str    = _snapshot_to_str(row.capital_after, flagged=bool(row.flags))
        meeting_date = row.date_of_shareholder_meeting or "On incorporation"
        meeting_type = row.meeting_type or "—"
        supp         = ", ".join(row.supporting_docs) if row.supporting_docs else "—"

        html_rows += f"""
        <tr class="{row_class}">
          <td>{meeting_date}</td>
          <td>{before_str}</td>
          <td>{after_str}</td>
          <td>{meeting_type}</td>
          <td><code>{row.source_document}</code></td>
          <td class="supp">{supp}</td>
          <td>{flag_cell}</td>
        </tr>"""

    output_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TechNova – Authorised Share Capital Change History</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px;
            background: #f7f8fc; color: #1a1a2e; }}
    .card {{ background: white; border-radius: 8px;
             box-shadow: 0 2px 12px rgba(0,0,0,0.08);
             padding: 36px 40px; max-width: 1350px; margin: 0 auto; }}
    h1 {{ font-size: 1.3em; font-weight: 700;
          border-bottom: 3px solid #16213e; padding-bottom: 10px; margin-bottom: 6px; }}
    .meta {{ font-size: 0.82em; color: #666; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.86em; }}
    thead th {{ background: #16213e; color: white;
                padding: 11px 13px; text-align: left; font-weight: 600; }}
    tbody td {{ padding: 11px 13px; border-bottom: 1px solid #e8eaf0;
                vertical-align: top; line-height: 1.5; }}
    tr.flagged td {{ background: #fffbf0; }}
    tr.clean   td {{ background: #f6fff8; }}
    tr:hover   td {{ filter: brightness(0.97); }}
    ul.flags {{ margin: 4px 0; padding: 0; list-style: none; }}
    ul.flags li {{ font-size: 0.81em; color: #b7410e; background: #fff3cd;
                   border-left: 3px solid #e67e22; padding: 4px 8px;
                   margin-bottom: 4px; border-radius: 0 3px 3px 0; }}
    .ok   {{ color: #155724; font-size: 0.85em; font-weight: 600; }}
    code  {{ background: #eef0ff; color: #3c3c8e; padding: 2px 6px;
             border-radius: 3px; font-size: 0.88em; }}
    .supp {{ color: #555; font-size: 0.82em; }}
    .legend {{ margin-top: 20px; padding: 12px 16px; background: #f0f4ff;
               border-radius: 6px; font-size: 0.81em; color: #444; }}
  </style>
</head>
<body>
<div class="card">
  <h1>TechNova Private Limited — Authorised Share Capital Change History</h1>
  <div class="meta">
    CIN: U72900MH2015PTC267801 &nbsp;|&nbsp;
    Generated by: DRHP Capital Structure Agent (LangGraph + Claude)
  </div>
  <table>
    <thead>
      <tr>
        <th>Date of Shareholder's Meeting</th>
        <th>Particulars of Change – From</th>
        <th>Particulars of Change – To</th>
        <th>AGM / EGM</th>
        <th>Source Document</th>
        <th>Supporting Docs</th>
        <th>Status / Flags</th>
      </tr>
    </thead>
    <tbody>{html_rows}
    </tbody>
  </table>
  <div class="legend">
    <strong>Legend:</strong>&nbsp;
    <span style="background:#f6fff8;padding:2px 8px;border-radius:3px;">
      Green = all fields confirmed across source documents
    </span>&nbsp;
    <span style="background:#fffbf0;padding:2px 8px;border-radius:3px;">
      Yellow = flagged — missing or conflicting information; field not filled by best guess
    </span>
    <br><br>
    Each row traces to a source SH-7 filing. All flags must be resolved before DRHP submission to SEBI.
  </div>
</div>
</body>
</html>"""

    return {"output_json": output_json, "output_html": output_html}
