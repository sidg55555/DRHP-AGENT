"""
state.py - Shared AgentState TypedDict for the DRHP Capital Structure Agent.

All LangGraph nodes read from and write to this state.
"""

from typing import TypedDict, Optional
from dataclasses import dataclass, field


@dataclass
class CapitalSnapshot:
    """Represents authorised capital at a point in time."""
    equity_shares: Optional[int] = None
    equity_face_value: Optional[float] = None
    equity_total: Optional[float] = None
    preference_shares: Optional[int] = None
    preference_face_value: Optional[float] = None
    preference_total: Optional[float] = None
    total_authorised: Optional[float] = None


@dataclass
class CapitalStructureRow:
    """One row in the final DRHP Authorised Share Capital change table."""
    date_of_shareholder_meeting: Optional[str]
    meeting_type: Optional[str]            # AGM / EGM / None (incorporation)
    capital_before: Optional[CapitalSnapshot]
    capital_after: CapitalSnapshot
    source_document: str                   # e.g. "SH7-002"
    supporting_docs: list[str]             # e.g. ["BR-002", "EGM-002", "MOA-002"]
    flags: list[str]                       # e.g. ["MISSING_EGM_RESOLUTION"]
    event: str                             # e.g. "Increase in Authorised Capital"


class AgentState(TypedDict):
    """Shared state flowing through every LangGraph node."""

    # --- Input ---
    raw_documents: list[dict]          # all loaded JSON dicts (SH-7s + attachments)

    # --- After classify_node ---
    classified_documents: list[dict]   # each doc annotated with confirmed type

    # --- After link_node ---
    filing_groups: list[dict]          # [{sh7: {...}, attachments: [...]}]

    # --- After extract_node ---
    extracted_data: list[dict]         # structured extraction per filing group

    # --- After validate_node ---
    validated_data: list[dict]         # extraction + flags for missing/conflicting fields

    # --- After table_builder_node ---
    capital_structure_rows: list[CapitalStructureRow]

    # --- After render_node ---
    output_json: dict
    output_html: str

    # --- Meta ---
    errors: list[str]                  # any processing errors (non-fatal)
