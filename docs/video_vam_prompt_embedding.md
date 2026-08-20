# Cosmos T5 prompt embedding artifact

The LeRobot-owned generator reproduces the official mimic-video T5 path from
commit `e3355dbc93132b576c02f920a59b4fc18a4f5906`:

- `T5TokenizerFast` and `T5EncoderModel` are loaded only from the verified local
  `t5-11b` directory with `local_files_only=True`.
- Transformers 5.5.4 may expose no callable `batch_encode_plus`; the generator
  calls it when available and otherwise calls the tokenizer object with the exact
  same prompts and kwargs. This compatibility choice is recorded in provenance
  as `batch_encode_plus_or_tokenizer_call_same_kwargs` and does not alter token IDs.
- `return_length=True` is retained for the official contract, but its returned
  value is ignored: true content lengths always come from `attention_mask.sum(dim=1)`.
- The fixed prompt is `take cube out of box` and the fixed padded sequence length
  is 512.
- The saved tensor is `[1, 512, 1024]` `bfloat16` on CPU, with a boolean
  attention mask. Positions after the mask length are exactly zero.
- The safetensors file has a JSON provenance sidecar. The loader verifies the
  tensor keys, shape, dtype, finite values, zero padding, sidecar schema, pinned
  model provenance, and output hash. No pickle is used.

## Generate and verify

The official model file is already present and verified at the path below. Run
this exact command from abakus only after reviewing the implementation:

```bash
cd /home/anton/lerobot-video-vam
.venv/bin/python scripts/video_vam/generate_cosmos_prompt_embedding.py \
  --model-path /home/anton/.cache/video-vam/mimic-video-f2833903/text_encoder/t5-11b \
  --output /home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors \
  --device auto
```

The default `auto` mode uses Accelerate `device_map="auto"`, a `20GiB` GPU
limit, a `40GiB` CPU limit, bf16 weights, and low-CPU-memory loading. It sends
`input_ids` to the actual T5 input-embedding device, which is required for a
sharded model. Use `--device cpu` only when the full bf16 model can fit in the
specified CPU limit. The generator refuses to overwrite either output file
unless `--overwrite` is explicit.

The generator performs its own round-trip comparison before reporting success.
A later, read-only verification can be run without loading T5:

```bash
cd /home/anton/lerobot-video-vam
.venv/bin/python -c 'from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding; a = load_prompt_embedding("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"); print(a.embedding.shape, a.embedding.dtype, a.provenance.output_sha256)'
```

`--skip-model-hash` is reserved for test doubles and still checks model file
presence and size. It must not be used for the official one-time generation.

## T5 retention

The current setup retains the downloaded 45 GB `pytorch_model.bin` alongside
its tokenizer and configuration files. The generated safetensors prompt
artifact is the runtime input for the extractor, while the retained T5 files
remain available for regeneration or independent verification. No deletion
step is part of this workflow.
