# Local installation

This repository is a source project, not a verified published PyPI package. Install from the Git checkout rather than relying on a package named `vehicle-detection` from PyPI.

## Requirements

- Python 3.11 is the current CI target.
- A virtual environment and Internet access to install the dependencies in `requirements.txt`.
- Model weights for the chosen YOLO detector; OCR may require additional downloads.

```bash
git clone https://github.com/BalajiKoushik01/whilter_ai_project_vscode.git
cd whilter_ai_project_vscode
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python main.py --help
```

The top-level `main.py` delegates to `vehicle_detection.main`. That implementation, rather than an invented CLI, is the reference for supported commands.

## Checks

```bash
python -m compileall -q vehicle_detection
python -m pytest tests/
```

The tests may be outdated relative to the detector's implementation. Record any failure with its traceback; do not interpret a successful dependency install as proof that the application works end-to-end.

## Notes

A model file such as `yolov8n.pt` may already be present in the repo; verify its origin and licence before redistribution. The output and input directories can contain vehicle registration identifiers and should not be used for confidential real-world data without appropriate permission and retention controls.
