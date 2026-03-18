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


class PlaybackScriptCue(BaseModel):
    cue_id: str
    entry_start: float = Field(ge=0.0)
    focus_start: float = Field(ge=0.0)
    focus_time: float = Field(ge=0.0)
    focus_end: float = Field(ge=0.0)
    exit_end: float = Field(ge=0.0)
    entry_rate: float = Field(gt=0.0)
    focus_rate: float = Field(gt=0.0)
    exit_rate: float = Field(gt=0.0)
    freeze_ms: int = Field(ge=0)


class CueArtifact(BaseModel):
    cue_id: str
    label: str
    url: str


class EvidenceArtifacts(BaseModel):
    normalized_source_url: Optional[str] = None
    annotated_replay_url: Optional[str] = None
    cue_clip_urls: List[CueArtifact] = Field(default_factory=list)
    cue_still_urls: List[CueArtifact] = Field(default_factory=list)
    frame_strip_urls: List[CueArtifact] = Field(default_factory=list)


class MediaProfile(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps: float = Field(ge=0.0)
    duration_s: float = Field(ge=0.0)
    frame_count: int = Field(ge=0)
    codec: Optional[str] = None
    bitrate: Optional[int] = Field(default=None, ge=0)
    rotation: int = 0
    detail_tier: str = Field(description="ultra_detail|high_detail|standard_detail")
    capture_assessment: str
    dense_refinement_used: bool = False


class AnalysisResult(BaseModel):
    phases: List[PhaseMetrics]
    issues: List[Issue]
    summary: AnalysisSummary
    coaching_plan: List[CoachingStep] = Field(default_factory=list)
    strengths: List[Strength] = Field(default_factory=list)
    research_notes: List[ResearchNote] = Field(default_factory=list)
    comparison: ComparisonResult
    playback_cues: List[PlaybackCue] = Field(default_factory=list)
    playback_script: List[PlaybackScriptCue] = Field(default_factory=list)
    confidence_notes: List[str] = Field(default_factory=list)
    media_profile: Optional[MediaProfile] = None
    artifacts: EvidenceArtifacts = Field(default_factory=EvidenceArtifacts)
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
    image_b64: Optional[str] = None
    image_url: Optional[str] = None


class PhaseBoundary(BaseModel):
    phase: str
    start: float
    end: float


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str = Field(description="completed|queued|failed")
    result: Optional[AnalysisResult] = None
    message: Optional[str] = None


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str = Field(description="queued|running|completed|failed")
    stage: str = Field(
        description="queued|inspecting|normalizing|coarse_scan|dense_refine|rendering|completed|failed"
    )
    progress_message: str
    media_profile: Optional[MediaProfile] = None
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
