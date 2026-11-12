# Vehicle and licence-plate detection prototype

This project explores detection and processing of vehicles and licence plates in images and video. The source includes a Python entry point, image-processing helpers, example scripts and tests.

**Status:** computer-vision prototype. Detection quality varies with camera conditions, vehicle orientation and the model weights used. This project is not an independently validated surveillance or identification system.

## Project layout

- `main.py` — compatibility entry point that delegates to `vehicle_detection.main`.
- `vehicle_detection/` — detection and processing modules (where included in your checkout).
- `examples/` — single-image and batch processing examples.
- `docs/` — installation, usage, developer guidance and troubleshooting.
- `tests/` — initial detector-related tests.
- `inputs/`, `outputs/` — local sample images and generated outputs.

## Setup

```bash
git clone https://github.com/BalajiKoushik01/whilter_ai_project_vscode.git
cd whilter_ai_project_vscode
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py --help
```

The main entry point imports `vehicle_detection.main`; verify that package is present in your checkout and that the command-line options match the implementation. Depending on the chosen workflow, Ultralytics model weights and OCR packages may be downloaded separately.

## Tests and limitations

```bash
python -m pytest tests/
```

The repository has tests, but this review did not execute them or measure detection accuracy. Do not assume that examples will work unchanged on every Python version or OS.

Always obtain permission for image/video processing and handle licence plates or identifiable people in accordance with applicable privacy requirements. This research software has not been assessed for high-stakes identification.
