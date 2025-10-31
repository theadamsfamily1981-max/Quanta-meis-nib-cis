from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator


class ExperimentSpec(BaseModel):
    id: str = Field(..., description="Unique experiment id (e.g., CIS-01).")
    category: str = Field(..., description="Top-level bucket (data, ood, selective, cis, meta, etc.).")
    title: str
    goal: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    experiment_id: str
    summary: str
    method: str
    code: str = Field(..., description="Standalone Python code block Claude proposes (no external I/O secrets).")
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

    @validator("code")
    def code_must_be_python(cls, v: str) -> str:
        assert "```python" in v, "Code must be fenced as ```python ...```"
        return v


def result_schema_markdown() -> str:
    return (
        "{\n"
        "  "experiment_id": "CIS-01",\n"
        "  "summary": "1-2 sentence recap",\n"
        "  "method": "approach details",\n"
        "  "code": "```python\n# self-contained code here\n```",\n"
        "  "metrics": {"accuracy": 0.76},\n"
        "  "artifacts": ["path/to/file"],\n"
        "  "risks": ["risk 1"],\n"
        "  "next_steps": ["step 1"]\n"
        "}"
    )
