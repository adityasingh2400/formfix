# Evaluation & Debugging

## Datasets
- Held-out pro clips and amateur-like clips; balanced across camera buckets and shot types.
- Include low-visibility/occlusion cases to test robustness.

## Metrics
- Pose/3D: PCK/OKS; fraction of high-confidence frames.
- Issue detection: precision/recall/F1 per issue; timing error recall.
- Latency: per-stage timings; p50/p90 end-to-end.
- Robustness: false-positive rate under low confidence; graceful degradation rate.

## Tooling
- Visual debugger: overlay joints, angles, issue markers; phase timeline; sequencing alignment score.
- Confidence surfacing: per-frame/issue confidence and visibility warnings.
- Regression harness: golden clips with expected issues; alert on drift.

cd /Users/aditya/Desktop/formfix
rm -rf .venv
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version   # should show 3.11.x
python -m pip install --upgrade pip
python -m pip install --force-reinstall -r backend/requirements.txt


(.venv) aditya@Adityas-MacBook-Pro formfix % curl -X POST -F "file=@/Users/aditya/Desktop/IMG_5640.mov;type=video/quicktime" -F "shot_type=free_throw" "http://127.0.0.1:8000/analyze"
{"job_id":"f1488719-2aa8-4e39-bbbc-9b25c1c77ead","status":"completed","result":{"phases":[{"name":"load","angles":{"knee_flexion":117.7879134460667,"hip_flexion":150.83640173843557,"elbow_angle":92.04193827307076,"shoulder_angle":42.47745687746661,"wrist_height":-0.6251679388051438},"timings":{"duration":0.6842767295597484},"confidence":0.9734964293462257},{"name":"set","angles":{"knee_flexion":81.4596922089245,"hip_flexion":142.36432124020325,"elbow_angle":104.23700420610838,"shoulder_angle":47.640479471548694,"wrist_height":-0.8576407740062317},"timings":{"duration":0.0},"confidence":0.976467082897822},{"name":"release","angles":{"knee_flexion":81.4596922089245,"hip_flexion":142.36432124020325,"elbow_angle":104.23700420610838,"shoulder_angle":47.640479471548694,"wrist_height":-0.8576407740062317},"timings":{"duration":0.0},"confidence":0.976467082897822},{"name":"follow_through","angles":{"knee_flexion":122.20497957670243,"hip_flexion":158.29546140360685,"elbow_angle":126.04105188640058,"shoulder_angle":104.62985695403586,"wrist_height":-1.814111965965364},"timings":{"duration":0.940880503144654},"confidence":0.9819086268544196}],"issues":[{"name":"Knee bend shallow","severity":"medium","delta":"+8–12° knee flexion at load","confidence":0.9783952151735626,"phase":"load"},{"name":"Low release height","severity":"medium","delta":"Improve upward drive before release","confidence":0.9783952151735626,"phase":"release"},{"name":"Rushed upper-body sequencing","severity":"medium","delta":"Delay wrist/arm lift until legs extend","confidence":0.9783952151735626,"phase":"release"}],"confidence_notes":[]},"message":"Analysis complete."}%                                                                                                                                                                               
(.venv) aditya@Adityas-MacBook-Pro formfix %