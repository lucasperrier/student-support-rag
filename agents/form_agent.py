from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .utils import BaseAgent

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


class FormAgent(BaseAgent):
    """
    Form agent: Handles structured form-filling for leads (name, email, interest).

    Behavior:
    - Extract name/email/interest when present
    - If missing required fields, ask for them
    - Persist to leads.json
    """

    def __init__(
        self,
        name: str,
        llm_client: Any,
        leads_path: Path = Path("data/leads.json"),
        tools: Optional[List[Dict[str, Any]]] = None,
        memory_path: Optional[str] = None,
    ):
        # NOTE: `tools` is accepted for backwards/interop compatibility with older tests/wiring.
        # This agent defines its own toolset internally; external tools are ignored.
        tools = [{"name": "save_lead", "description": "Save a lead to file", "function": self._save_lead}]
        super().__init__(name, llm_client, tools, memory_path=None)

        self.leads_path = leads_path
        self.leads_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.leads_path.exists():
            self.leads_path.write_text("[]", encoding="utf-8")

    def get_system_prompt(self) -> str:
        return (
            "You are the form agent for ESILV admissions. Your job is to collect name, email, and interest. "
            "If information is missing, ask politely and clearly. Once name and email are provided, save the lead."
        )

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        name = self._extract_name(query)
        email = self._extract_email(query)
        interest = self._extract_interest(query)

        missing: List[str] = []
        if not name:
            missing.append("full name")
        if not email:
            missing.append("email address")

        if missing:
            return {
                "answer": f"To help with admissions, I need your {', '.join(missing)}. Please provide it.",
                "sources": [],
                "action": "collect_lead",
            }

        lead = {
            "id": f"lead-{int(time.time() * 1000)}",
            "name": name,
            "email": email,
            "interest": interest or "",
        }
        self.call_tool("save_lead", {"lead": lead})
        self.log_action(f"Saved lead: {name} <{email}>")

        return {
            "answer": f"Thanks, {name}! I saved your contact info. {('Interest: ' + interest) if interest else ''}".strip(),
            "sources": [],
            "action": "lead_saved",
        }

    def _extract_email(self, text: str) -> Optional[str]:
        m = EMAIL_RE.search(text)
        return m.group(0) if m else None

    def _extract_name(self, text: str) -> Optional[str]:
        """
        Heuristic name extraction:
        - "my name is John Doe" => John Doe
        - "I am John Doe" => John Doe
        - Otherwise: None (avoid extracting a random word)
        """
        patterns = [
            r"\bmy name is\s+([A-Za-zÀ-ÖØ-öø-ÿ' -]{2,})",
            r"\bi am\s+([A-Za-zÀ-ÖØ-öø-ÿ' -]{2,})",
            r"\bje m'appelle\s+([A-Za-zÀ-ÖØ-öø-ÿ' -]{2,})",
            r"\bje suis\s+([A-Za-zÀ-ÖØ-öø-ÿ' -]{2,})",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # Stop at punctuation that usually ends the name segment
                candidate = re.split(r"[.,;:!?\n\r\t]", candidate)[0].strip()
                # Stop at connector words that signal the name has ended
                # (e.g. "Jane Doe and my email is ..." -> "Jane Doe")
                candidate = re.split(
                    r"\b(?:and|et|my|je|suis|i'?m|interested|interest|email|e-?mail|is|with|from|the|at|but)\b",
                    candidate,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()
                # Names are short: keep at most the first 4 tokens
                candidate = " ".join(candidate.split()[:4]).strip()
                # Avoid capturing very long strings
                if 2 <= len(candidate) <= 60:
                    return candidate
        return None

    def _extract_interest(self, text: str) -> Optional[str]:
        """
        Heuristic interest extraction:
        - "I'm interested in X"
        - "interest: X"
        """
        patterns = [
            r"\binterested in\s+(.+)$",
            r"\binterest\s*:\s*(.+)$",
            r"\bje suis intéressé(?:e)? par\s+(.+)$",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # End the interest at a sentence break or list separator, and drop any email
                candidate = re.split(r"[.,\n\r]", candidate)[0].strip()
                candidate = EMAIL_RE.sub("", candidate).strip()
                if candidate:
                    return candidate[:120]
        return None

    def _save_lead(self, lead: Dict[str, Any]) -> None:
        leads = self._load_leads()
        leads.append(lead)
        self.leads_path.write_text(json.dumps(leads, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_leads(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.leads_path.read_text(encoding="utf-8"))
        except Exception:
            return []