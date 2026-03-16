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


class AnalysisSummary(BaseModel):
    label: str
    headline: str
    start_here: str
    encouragement: str
    tracking_quality: str


class CoachingStep(BaseModel):
    title: str
    cue: str
    why: str
    drill: str
    phase: Optional[str] = None
    priority: str = Field(description="now|next")
    confidence: float = Field(ge=0.0, le=1.0)


class Strength(BaseModel):
    title: str
    detail: str


class ResearchNote(BaseModel):
    title: str
    body: str


class ComparisonTrait(BaseModel):
    title: str
    detail: str
    cue: Optional[str] = None
    phase: Optional[str] = None


class ComparisonMetricDelta(BaseModel):
    key: str
    label: str
    user_value: Optional[float] = None
    reference_value: Optional[float] = None
    unit: str = ""
    direction: str = Field(description="higher|lower|aligned")
    summary: str
    cue: Optional[str] = None


class ComparisonMatch(BaseModel):
    profile_id: str
    label: str
    category: str = Field(description="archetype|player")
    source: str = Field(description="seeded|trained")
    fit_score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    description: str
    style_tags: List[str] = Field(default_factory=list)
    matched_traits: List[ComparisonTrait] = Field(default_factory=list)
    borrow_traits: List[ComparisonTrait] = Field(default_factory=list)
    metric_deltas: List[ComparisonMetricDelta] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    status: str = Field(description="unavailable|seeded_reference|trained_reference")
    overview: str
    primary: Optional[ComparisonMatch] = None
    alternatives: List[ComparisonMatch] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class PlaybackAnchor(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    direction: str = Field(description="up|down|left|right|down_left|down_right")
    motion_type: str = Field(
        default="linear",
        description="linear|rotate_cw|rotate_ccw|arc_up|arc_down"
    )
    pivot_x: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="For rotation: center point x (normalized). Defaults to anchor position."
    )
    pivot_y: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="For rotation: center point y (normalized). Defaults to anchor position."
    )
    arc_radius: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.5,
        description="For arc/rotation: radius as fraction of frame width"
    )
    magnitude: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="How much motion to show (0.5=subtle, 1.0=normal, 2.0=large)"
    )


class ResearchStat(BaseModel):
    label: str
    value: str
    detail: str


class ResearchSource(BaseModel):
    label: str
    detail: str
    url: Optional[str] = None


class PlaybackDeepDive(BaseModel):
    summary: str
    stats: List[ResearchStat] = Field(default_factory=list)
    sources: List[ResearchSource] = Field(default_factory=list)


class PlaybackCue(BaseModel):
    id: str
    title: str
    cue: str
    why: str
    research_basis: str
    deep_dive: Optional[PlaybackDeepDive] = None
    phase: Optional[str] = None
    timestamp: float = Field(ge=0.0)
    bubble_x: float = Field(ge=0.0, le=1.0)
    bubble_y: float = Field(ge=0.0, le=1.0)
    anchors: List[PlaybackAnchor] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    phases: List[PhaseMetrics]
    issues: List[Issue]
    summary: AnalysisSummary
    coaching_plan: List[CoachingStep] = Field(default_factory=list)
    strengths: List[Strength] = Field(default_factory=list)
    research_notes: List[ResearchNote] = Field(default_factory=list)
    comparison: ComparisonResult
    playback_cues: List[PlaybackCue] = Field(default_factory=list)
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
    processing_mode: str = Field(description="quick|deep")


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
