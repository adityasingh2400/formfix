import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import schemas
from .services.job_manager import AnalysisJobManager
from .services.media_store import LocalMediaStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Basketball Form Analyzer", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

media_store = LocalMediaStore()
job_manager = AnalysisJobManager(media_store)
app.mount("/media", StaticFiles(directory=str(media_store.root)), name="media")


def _normalize_shooting_hand(value: str | None) -> str | None:
    normalized = value.strip().lower() if value else None
    if normalized not in {None, "left", "right"}:
        raise HTTPException(status_code=400, detail="Shooting hand must be 'left' or 'right'.")
    return normalized


def _validate_upload(file: UploadFile) -> None:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Upload must be a video file.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analysis-jobs", response_model=schemas.AnalysisJobResponse, status_code=202)
async def create_analysis_job(
    file: UploadFile = File(...),
    shooting_hand: str | None = Form(default=None),
    shot_type: str | None = Form(default=None),
):
    _validate_upload(file)
    normalized_shooting_hand = _normalize_shooting_hand(shooting_hand)
    content = await file.read()

    try:
        job = job_manager.create_job(
            file_bytes=content,
            filename=file.filename or "upload.mp4",
            shot_type=shot_type,
            shooting_hand=normalized_shooting_hand,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not queue analysis: {exc}") from exc

    return JSONResponse(status_code=202, content=job.model_dump())


@app.get("/analysis-jobs/{job_id}", response_model=schemas.AnalysisJobResponse)
def get_analysis_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    return JSONResponse(content=job.model_dump())


@app.post("/analyze", response_model=schemas.AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    shooting_hand: str | None = Form(default=None),
    shot_type: str | None = Form(default=None),
    return_visuals: bool = Form(default=True),
):
    _validate_upload(file)
    normalized_shooting_hand = _normalize_shooting_hand(shooting_hand)
    content = await file.read()

    logger.info(
        "Analyze request: shooting_hand=%s, normalized=%s, shot_type=%s",
        shooting_hand,
        normalized_shooting_hand,
        shot_type,
    )

    try:
        job_id, result = job_manager.run_inline_analysis(
            file_bytes=content,
            filename=file.filename or "upload.mp4",
            shot_type=shot_type,
            shooting_hand=normalized_shooting_hand,
        )
        if not return_visuals:
            result = result.model_copy(
                update={
                    "artifacts": result.artifacts.model_copy(update={"annotated_replay_url": None}),
                    "keyframes": [],
                    "annotated_video_b64": None,
                    "processing_mode": "quick",
                }
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return JSONResponse(
        content=schemas.AnalyzeResponse(
            job_id=job_id,
            status="completed",
            result=result,
            message="Analysis complete.",
        ).model_dump()
    )
