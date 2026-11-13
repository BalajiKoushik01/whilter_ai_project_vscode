# Development and test isolation

The current CI pipeline installs requirements, performs Python lint checks and executes the detector test suite. Detector accuracy and licence-plate recognition quality are not measured by these unit tests.

Tests should use `pytest` fixtures such as `tmp_path` for temporary model or image inputs. Do not generate files inside the repository or depend on previously tracked output samples.

## Recommended acceptance checks
1. Verify clean installation on the documented Python version and run `pytest`.
2. Test valid/missing/corrupted model weights and unreadable images.
3. Establish licensed sample datasets across daylight, night, partial occlusion, perspective, motion blur and multiple vehicle classes.
4. Record precision, recall, false positives, processing latency, hardware, model hash and input dimensions.
5. Test image and video outputs for unintended exposure of identifying information and implement retention where necessary.
6. Validate generated logs and sample exports never contain real personal records or secrets.

Existing `.pt` model artifacts and samples are preserved until their origin, licence and reproducible acquisition are documented. Adding ignore rules does not remove already tracked files.
