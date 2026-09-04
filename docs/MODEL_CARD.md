# Model Card — AlphaTiger (browser ONNX)

**Status:** Stub for expansion track (no weight cleanup in this iter)

| Artifact | Path | Notes |
|----------|------|-------|
| Browser runtime weights | `public/alphatiger.onnx` | Loaded by `onnxruntime-web` in the static client |
| Exported twin (repo) | `ai/models/alphatiger.onnx` | Same bytes as public copy when last synced |
| Canonical PyTorch source | `ai/models/alphatigerv13.pt` | `DEFAULT_MODEL` in `game/constants.py`; consumed by `ai/onnx.py` |
| Export command | `python -m ai.onnx` then `cp ai/models/alphatiger.onnx public/alphatiger.onnx` | See README “Exporting PyTorch Checkpoints to ONNX” |

## Policy

- Do **not** mass-delete historical `alphatigerv*.pt` checkpoints in this track.
- Keep v13 (+ betas) as the documented production lineage until a dedicated hygiene PR decides otherwise.
- Training still writes under `checkpoints/`; browser play does not load `.pt` files.
