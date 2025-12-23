from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import schemas
from .services import analyzer

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
    shot_type: str | None = None,
    return_visuals: bool = True,
):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Upload must be a video file.")

    content = await file.read()
    job_id = analyzer.new_job_id()
    try:
        result = analyzer.analyze_video(content, shot_type=shot_type, return_visuals=return_visuals)
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

