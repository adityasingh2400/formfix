# System Architecture & Infra

## Services
- API gateway for uploads/results.
- GPU inference worker for pose + models.
- Job queue (Redis/Cloud Tasks) for async processing.
- Object storage (S3/GCS) for raw/processed media.
- Postgres for metadata and labels.

## Tech stack
- Python for pose/inference (PyTorch/ONNXRuntime).
- Node/Go for API/orchestration.
- Structured logging + per-stage timing; pose-quality flags.

## Observability & ops
- Metrics: pose success rate, issue detection rates, latency p50/p90, failure codes.
- Alerts on pose-confidence collapse, queue backlog, latency SLO breaches.
- Retention controls and delete-on-complete paths for user media.

