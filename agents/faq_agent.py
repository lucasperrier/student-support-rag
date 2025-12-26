from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .utils import BaseAgent


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class FAQAgent(BaseAgent):
    def __init__(self, name: str, llm_client: Any, faq_path: Optional[str] = None):
        super().__init__(name, llm_client, tools=[], memory_path=None)
        self.faq_path = faq_path
        self.faqs = self._load_faqs()  # Dict of {"question": "answer"}

        # Precompute normalized keys for matching
        self._faq_items: List[Tuple[str, str, str]] = [
            (q, _normalize(q), a) for q, a in self.faqs.items()
        ]

    def _load_faqs(self) -> Dict[str, str]:
        if self.faq_path and Path(self.faq_path).exists():
            try:
                return json.loads(Path(self.faq_path).read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "what is esilv": "ESILV is an engineering school in France. For official details and the latest information, refer to ESILV documents and website.",
            "how to apply": "To apply, visit the ESILV website and submit your application online with the required documents.",
            "admission requirements": "Admission requirements depend on your profile and program. If you want, I can search the official documents for the exact requirements.",
            "courses offered": "ESILV offers courses in AI, cybersecurity, big data, and more. For an accurate list, I can search the official program documents.",
            "contact information": "You can contact ESILV via official channels on the website. If you’re applying, you can also leave your name/email and I’ll save it.",
        }

    def get_system_prompt(self) -> str:
        return (
            "You are an FAQ agent. Answer only if the question matches a known FAQ. "
            "Otherwise, suggest using the retrieval agent."
        )

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        qn = _normalize(query)
        q_tokens = set(_tokenize(qn))

        best = None  # (score, original_question, answer)
        for original_q, norm_q, answer in self._faq_items:
            # Exact / substring match gets priority
            if norm_q == qn or norm_q in qn or qn in norm_q:
                best = (1.0, original_q, answer)
                break

            # Soft token overlap score
            faq_tokens = set(_tokenize(norm_q))
            if not faq_tokens:
                continue
            overlap = len(q_tokens & faq_tokens) / max(1, len(faq_tokens))
            if best is None or overlap > best[0]:
                best = (overlap, original_q, answer)

        if best and best[0] >= 0.6:
            _, matched_q, answer = best
            return {
                "answer": answer,
                "sources": [{"id": "faq", "meta": {"question": matched_q}}],
                "action": "answer",
            }

        return {
            "answer": "I don’t have that in my FAQ. Try asking your question again, or ask about a specific ESILV document topic.",
            "sources": [],
            "action": "answer",
        }