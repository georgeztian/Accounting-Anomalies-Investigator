"""Structured-output schemas for the investigation agent's LLM nodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ToolName = Literal["get_fact", "ratio", "peer_stats", "search_filing", "explaining_text", "get_events", "diff_risk_factors"]


class Hypothesis(BaseModel):
    statement: str = Field(description="One sentence: what reporting choice could produce these signals")
    confidence: float = Field(ge=0, le=1)
    innocent_alternatives: list[str] = Field(description="Legitimate business or accounting explanations to test")


class Hypotheses(BaseModel):
    hypotheses: list[Hypothesis]


class EvidenceTask(BaseModel):
    hypothesis_index: int = Field(ge=0)
    tool: ToolName
    purpose: str
    item: str | None = Field(default=None, description="Canonical line item id for get_fact / explaining_text, e.g. AR")
    numerator: str | None = Field(default=None, description="Canonical item id for ratio")
    denominator: str | None = Field(default=None, description="Canonical item id for ratio")
    ratio_key: str | None = Field(default=None, description="Ratio key for peer_stats")
    query: str | None = Field(default=None, description="Keywords for search_filing (plain words, no regex)")
    section: Literal["item1", "item1a", "item7", "item8", "item9a", "any"] | None = None
    event_types: list[str] | None = None


class EvidencePlan(BaseModel):
    tasks: list[EvidenceTask]


class AlternativeAssessment(BaseModel):
    alternative: str
    supported: bool = Field(description="True only if ledger evidence clearly supports this innocent explanation")
    strength: Literal["weak", "moderate", "strong"]
    ledger_ids: list[str]
    reasoning: str


class AlternativeAssessments(BaseModel):
    assessments: list[AlternativeAssessment]


class AlternativeLine(BaseModel):
    alternative: str
    assessment: str


class FindingNarrative(BaseModel):
    summary: str = Field(description="2-4 sentences: what the numbers show and why it matters")
    explanation: str = Field(description="One paragraph walking through the evidence, citing ledger ids like [L3]")
    alternatives: list[AlternativeLine]
    questions: list[str]
    cited_ledger_ids: list[str]
