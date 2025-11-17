"""
Pydantic models for T-FAN API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class MetricsResponse(BaseModel):
    """Current training metrics."""
    training_active: bool = False
    step: int = 0
    accuracy: float = 0.0
    latency_ms: float = 0.0
    hypervolume: float = 0.0
    epr_cv: float = 0.0
    topo_gap: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ParetoWeights(BaseModel):
    """Pareto optimization decision weights."""
    neg_accuracy: float = Field(10.0, description="Weight for accuracy (higher = maximize)")
    latency: float = Field(1.0, description="Weight for latency (higher = minimize)")
    epr_cv: float = Field(2.0, description="Weight for EPR CV (higher = minimize)")
    topo_gap: float = Field(1.0, description="Weight for topology gap (higher = minimize)")
    energy: float = Field(0.5, description="Weight for energy (higher = minimize)")


class TrainingRequest(BaseModel):
    """Training start request."""
    config_path: str = Field("configs/auto/best.yaml", description="Path to training config")
    max_steps: Optional[int] = Field(None, description="Max training steps")
    logdir: str = Field("runs/api_training", description="Log directory")


class TrainingStatus(BaseModel):
    """Training status response."""
    active: bool
    step: int = 0
    config: Optional[str] = None
    started_at: Optional[str] = None


class ConfigResponse(BaseModel):
    """Configuration response."""
    path: str
    name: str
    config: Dict


class ParetoConfig(BaseModel):
    """Pareto configuration point."""
    n_heads: int
    d_model: int
    n_layers: int
    keep_ratio: float
    alpha: float
    lr: float
    objectives: List[float]


class ParetoFrontResponse(BaseModel):
    """Pareto front response."""
    n_pareto_points: int
    hypervolume: float
    configurations: List[ParetoConfig]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
