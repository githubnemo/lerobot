import torch


def test_precompute_creates_nested_output_parent_before_progress(monkeypatch, tmp_path):
    from scripts.video_vam import precompute_fastwam_text as module

    output = tmp_path / "nested" / "fastwam" / "contexts.safetensors"
    monkeypatch.setattr(module, "_dataset_tasks", lambda args: ["pick up the cube"])
    monkeypatch.setattr(module, "build_wan_tokenizer", lambda **kwargs: object())
    monkeypatch.setattr(module, "load_pretrained_wan_text_encoder", lambda **kwargs: object())

    def encode(*, tokenizer, text_encoder, prompts, device):
        del tokenizer, text_encoder, device
        return torch.zeros((len(prompts), 128, 4096)), torch.ones((len(prompts), 128), dtype=torch.bool)

    monkeypatch.setattr(module, "encode_wan_text_context", encode)
    monkeypatch.setattr(
        module,
        "save_text_context_artifact",
        lambda path, context, context_mask, prompts, provenance: path.write_bytes(b"complete"),
    )

    result = module.main(
        [
            "--output",
            str(output),
            "--dtype",
            "float32",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert output.parent.is_dir()
    assert output.is_file()
