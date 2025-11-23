import json
from pathlib import Path
from typing import Dict, Any, Optional
from .utils import BaseAgent

class FAQAgent(BaseAgent):
    def __init__(self, name: str, llm_client: Any, faq_path: Optional[str] = None):
        super().__init__(name, llm_client, tools=[], memory_path=None)
        self.faq_path = faq_path
        self.faqs = self._load_faqs()  # Dict of {"question": "answer"}

    def _load_faqs(self) -> Dict[str, str]:
        """Load static FAQs from file or use defaults."""
        if self.faq_path and Path(self.faq_path).exists():
            try:
                return json.loads(Path(self.faq_path).read_text())
            except Exception:
                pass
        # Default FAQs for ESILV
        return {
            "what is esilv": "ESILV is a leading engineering school in France, offering programs in computer science, data, and innovation.",
            "how to apply": "To apply, visit the ESILV website and submit your application online with required documents.",
            "admission requirements": "Requirements include a high school diploma, language proficiency, and passing entrance exams.",
            "courses offered": "ESILV offers courses in AI, cybersecurity, big data, and more.",
            "contact information": "Contact ESILV at admissions@esilv.fr or visit their website."
        }

    def get_system_prompt(self) -> str:
        return "You are an FAQ agent. Answer common questions about ESILV using static knowledge. If unsure, suggest asking the retrieval agent."

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        # Simple exact or substring match
        for q, a in self.faqs.items():
            if q.lower() in query_lower or query_lower in q.lower():
                return {"answer": a, "sources": [{"id": "faq", "meta": {"question": q}}], "action": "answer"}
        # No match
        return {"answer": "I'm sorry, I don't have an answer for that in my FAQ. Try rephrasing or ask about ESILV programs.", "sources": [], "action": "answer"}