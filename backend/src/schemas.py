from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Issue(BaseModel):
    name: str
    severity: str = Field(description="low|medium|high")
    delta: Optional[str] = Field(
        default=None, description="Suggested change, e.g., +5 deg knee flexion"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    phase: Optional[str] = None


class PhaseMetrics(BaseModel):
    name: str
    angles: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Per-phase kinematic features (degrees for angles; may be null if landmarks are missing).",
    )
    timings: Dict[str, float] = Field(
        default_factory=dict, description="Timing metrics per phase in seconds"
    )
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    phases: List[PhaseMetrics]
    issues: List[Issue]
    confidence_notes: List[str] = Field(default_factory=list)
    annotated_video_b64: Optional[str] = Field(
        default=None, description="Base64 data URL of annotated video"
    )
    keyframes: List["KeyframePreview"] = Field(
        default_factory=list,
        description="Annotated keyframes per phase",
    )
    phase_boundaries: List["PhaseBoundary"] = Field(
        default_factory=list,
        description="Phase time windows in seconds",
    )


class KeyframePreview(BaseModel):
    phase: str
    image_b64: str


class PhaseBoundary(BaseModel):
    phase: str
    start: float
    end: float


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str = Field(description="completed|queued|failed")
    result: Optional[AnalysisResult] = None
    message: Optional[str] = None

