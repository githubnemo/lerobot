# Minimal Cosmos-Predict2 source closure

The files here are copied source, not a replacement implementation. Use the
LeRobot-owned `cosmos_predict2_extractor.py` wrapper for configuration,
checkpoint validation, preprocessing, and output metadata. The selected
official 2B backend is `minimal_a2a`, with upstream PyTorch SDPA dispatch in
`module/attention.py`; Transformer Engine is required lazily for the official normalization and fused rotary kernels; it is not the selected attention backend.
