"""Expected answer type detection from the question form (rule based)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_YESNO = re.compile(
    r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should|shall|may|might)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class AnswerType:
    kind: str  # yesno | person | date | location | number | organization | work | entity | other
    ner_labels: frozenset[str]


def expected_answer_type(question: str) -> AnswerType:
    q = " ".join(question.lower().split())
    if _YESNO.match(q) and " or " not in q:
        return AnswerType("yesno", frozenset())
    if re.search(r"\b(how many|how much|what number|what percentage|how old|how long|how far)\b", q):
        return AnswerType("number", frozenset({"CARDINAL", "QUANTITY", "MONEY", "PERCENT", "DATE"}))
    if re.search(r"\b(when|what year|which year|what date|which date|what decade|which century)\b", q):
        return AnswerType("date", frozenset({"DATE"}))
    if re.search(r"\b(where|which city|what city|which country|what country|which state|what state|"
                 r"which town|what town|which place|what place|which region|what region|which continent)\b", q):
        return AnswerType("location", frozenset({"GPE", "LOC", "FAC"}))
    if re.search(r"\b(which|what) (university|college|school|institute|institution|company|organization|"
                 r"organisation|team|club|band|firm|agency|lab|laboratory|party|network|studio|corporation)\b", q):
        return AnswerType("organization", frozenset({"ORG"}))
    if re.search(r"\b(which|what) (film|movie|book|novel|song|album|series|show|play|poem|painting|game|"
                 r"paper|award|prize)\b", q):
        return AnswerType("work", frozenset({"WORK_OF_ART", "EVENT", "ORG", "PRODUCT", "LAW"}))
    if re.search(r"^(who|whom)\b|\b(which|what) (person|scientist|actor|actress|author|writer|director|player|"
                 r"singer|politician|president|king|queen|physicist|chemist|researcher|inventor|founder|"
                 r"laureate|musician|artist|leader|ceo|coach)\b|\bwho\b", q):
        return AnswerType("person", frozenset({"PERSON", "ORG", "NORP"}))
    if q.startswith(("which", "what")):
        return AnswerType("entity", frozenset({"PERSON", "ORG", "GPE", "LOC", "FAC", "WORK_OF_ART", "EVENT",
                                               "NORP", "PRODUCT", "LAW", "LANGUAGE", "DATE"}))
    return AnswerType("other", frozenset())
