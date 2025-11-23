# Collects name, email, interest
from typing import Dict, Any, Optional
import json
from pathlib import Path
from .utils import BaseAgent

class FormAgent(BaseAgent):
    """
    Form agent: Handles structured form-filling for leads (name, email, interest).
    Collects user info and stores in leads.json.
    """

    def __init__(self, name: str, llm_client: Any, leads_path: Path = Path("data/leads.json")):
        tools = [
            {"name": "save_lead", "description": "Save a lead to file", "function": self._save_lead}
        ]
        super().__init__(name, llm_client, tools, memory_path=None)  # No persistent memory needed
        self.leads_path = leads_path
        self.leads_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.leads_path.exists():
            self.leads_path.write_text("[]")

    def get_system_prompt(self) -> str:
        return (
            "You are the form agent for ESILV admissions. Collect name, email, and interest. "
            "If info is missing, prompt politely. Once collected, save and confirm."
        )

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process form-related queries: Extract or prompt for name, email, interest.
        :param query: User message (may contain form data).
        :param context: Optional context.
        :return: Response dict with answer, action.
        """
        # Simple extraction (stub; replace with NLP for robustness)
        name = self._extract_field(query, ["name", "my name is"])
        email = self._extract_field(query, ["email", "@"])
        interest = self._extract_field(query, ["interest", "interested in"])

        if name and email:
            # Save lead
            lead = {"id": f"lead-{len(self._load_leads()) + 1}", "name": name, "email": email, "interest": interest or ""}
            self.call_tool("save_lead", {"lead": lead})
            self.log_action(f"Saved lead: {name}")
            return {
                "answer": f"Thanks, {name}! Your contact info has been saved. We'll reach out about {interest or 'ESILV programs'}.",
                "sources": [],
                "action": "lead_saved"
            }
        else:
            # Prompt for missing info
            missing = []
            if not name:
                missing.append("full name")
            if not email:
                missing.append("email address")
            prompt = f"I need your {', '.join(missing)} to help with admissions. Please provide them."
            return {
                "answer": prompt,
                "sources": [],
                "action": "collect_lead"
            }

    def _extract_field(self, text: str, keywords: list) -> Optional[str]:
        """Simple keyword-based extraction (stub)."""
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                # Naive: take next word; improve with regex/NLP
                parts = text.split()
                idx = next((i for i, p in enumerate(parts) if kw.lower() in p.lower()), -1)
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        return None

    def _save_lead(self, lead: Dict[str, Any]):
        """Tool: Save lead to JSON file."""
        leads = self._load_leads()
        leads.append(lead)
        self.leads_path.write_text(json.dumps(leads, ensure_ascii=False, indent=2))

    def _load_leads(self) -> list:
        """Load leads from file."""
        try:
            return json.loads(self.leads_path.read_text())
        except Exception:
            return []