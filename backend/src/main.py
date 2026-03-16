from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from . import schemas
from .services import analyzer

logger = logging.getLogger(__name__)

app = FastAPI(title="Basketball Form Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=schemas.AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    shooting_hand: str | None = Form(default=None),
    shot_type: str | None = Form(default=None),
    return_visuals: bool = Form(default=True),
):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Upload must be a video file.")

    normalized_shooting_hand = shooting_hand.strip().lower() if shooting_hand else None
    if normalized_shooting_hand not in {None, "left", "right"}:
        raise HTTPException(status_code=400, detail="Shooting hand must be 'left' or 'right'.")

    logger.info(
        "Analyze request: shooting_hand=%s, normalized=%s, shot_type=%s",
        shooting_hand, normalized_shooting_hand, shot_type
    )

    content = await file.read()
    job_id = analyzer.new_job_id()
    try:
        result = analyzer.analyze_video(
            content,
            shot_type=shot_type,
            shooting_hand=normalized_shooting_hand,
            return_visuals=return_visuals,
        )
    except Exception as exc:  # pragma: no cover - placeholder safety
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return JSONResponse(
        content=schemas.AnalyzeResponse(
            job_id=job_id,
            status="completed",
            result=result,
            message="Analysis complete.",
        ).model_dump()
    )
