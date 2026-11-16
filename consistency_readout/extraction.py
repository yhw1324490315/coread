"""Generate full-width prompt representations from a local Qwen2 model."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os

import numpy as np


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    records, seen = [], set()
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            sid = row.get("sample_id")
            if not isinstance(sid, str) or not sid.strip() or sid in seen:
                raise ValueError(f"Invalid or duplicate sample_id at prompt line {line_number}")
            rendered, text = row.get("rendered_prompt"), row.get("text")
            if rendered is not None and (not isinstance(rendered, str) or not rendered.strip()):
                raise ValueError(f"Invalid rendered_prompt at line {line_number}")
            if rendered is None and (not isinstance(text, str) or not text.strip()):
                raise ValueError(f"Missing prompt text at line {line_number}")
            records.append({"sample_id": sid, "rendered_prompt": rendered, "text": text})
            seen.add(sid)
    if not records:
        raise ValueError("Prompt file is empty")
    return records


def _model_files(path, report):
    files = sorted(p for p in path.rglob("*") if p.is_file()
                   and not any(part.startswith(".") for part in p.relative_to(path).parts)
                   and (p.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja"}
                        or p.name in {"merges.txt", "vocab.txt"}))
    if not files:
        raise ValueError("Local model directory has no model/tokenizer files")
    result = {}
    for i, file in enumerate(files, 1):
        report(f"model identity: hash file {i}/{len(files)} ({file.name})")
        result[file.relative_to(path).as_posix()] = _sha(file)
    return result


def _load_model(path, device, dtype):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ImportError("Feature extraction needs optional dependencies: python -m pip install '.[extract]'") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu or install a CUDA-enabled PyTorch")
    if device == "cpu" and dtype != "float32":
        raise ValueError("CPU extraction uses float32; select --dtype float32")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or eos token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        dtype=getattr(torch, dtype), attn_implementation="sdpa")
    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError("This extraction protocol supports local Qwen2/Qwen2.5 checkpoints")
    model.to(device)
    model.eval()
    return tokenizer, model, torch


def _batch_vectors(records, tokenizer, model, torch, device):
    texts = [row["rendered_prompt"] if row["rendered_prompt"] is not None else
             tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}],
                                           tokenize=False, add_generation_prompt=True)
             for row in records]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False,
                        add_special_tokens=False, return_attention_mask=True)
    context = int(model.config.max_position_embeddings)
    if encoded["input_ids"].shape[1] > context:
        raise ValueError(f"Prompt exceeds the model context limit ({context}); no truncation is performed")
    encoded = {key: value.to(device) for key, value in encoded.items()
               if key in {"input_ids", "attention_mask"}}
    mask = encoded["attention_mask"]
    if not bool(mask.any(dim=1).all()):
        raise ValueError("A prompt tokenized to an empty sequence")
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    with torch.inference_mode():
        hidden = model(**encoded, position_ids=positions, use_cache=False,
                       return_dict=True).last_hidden_state
        offsets = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = offsets.masked_fill(mask == 0, -1).max(dim=1).values
        vectors = hidden[torch.arange(len(records), device=hidden.device), last].float().cpu().numpy()
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Model returned invalid final prompt representations")
    return vectors


def _write_manifest(path, manifest):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _validate_resume(manifest, configuration, output, records, names):
    if manifest.get("schema_version") != 1 or manifest.get("configuration") != configuration:
        raise ValueError("Extraction configuration changed; choose a new output directory")
    completed = 0
    chunk_records = manifest.get("chunks", [])
    if not isinstance(chunk_records, list):
        raise ValueError("Invalid feature checkpoint manifest")
    for i, chunk in enumerate(chunk_records):
        if i >= len(names) or chunk.get("file") != names[i] or chunk.get("start") != completed:
            raise ValueError("Feature checkpoint chunks are not contiguous")
        path = output / names[i]
        expected_rows = min(configuration["chunk_size"], len(records) - completed)
        if chunk.get("rows") != expected_rows or not path.is_file() or _sha(path) != chunk.get("sha256"):
            raise ValueError("Feature checkpoint file/count/hash differs")
        with np.load(path, allow_pickle=False) as bundle:
            ids, features = bundle["sample_id"], bundle["features"]
        expected_ids = np.asarray([row["sample_id"] for row in records[completed:completed + expected_rows]])
        if (not np.array_equal(ids, expected_ids) or features.dtype != np.float32
                or features.shape != (expected_rows, manifest.get("feature_dimension"))
                or not np.isfinite(features).all()):
            raise ValueError("Feature checkpoint IDs or representation schema differs")
        completed += expected_rows
    observed = {p.name for p in output.glob("features*.npz")}
    if observed != {row["file"] for row in chunk_records} or manifest.get("completed_rows") != completed:
        raise ValueError("Uncheckpointed feature files or inconsistent completed count; choose a new output directory")
    if manifest.get("status") not in {"running", "failed", "complete"}:
        raise ValueError("Invalid feature checkpoint status")
    if manifest.get("status") == "complete" and completed != len(records):
        raise ValueError("Completed feature checkpoint has incomplete prompt coverage")
    return completed


def extract_features(data_dir, model_path, output_dir=None, batch_size=2, chunk_size=6000,
                     device="cuda", dtype=None, resume=False, progress=None):
    """Generate local features from a dataset directory or prompts JSONL(.gz).

    No model is downloaded. Completed chunks are checkpointed; ``resume`` only
    reuses files matching the model, prompts, runtime, configuration, IDs and hashes.
    Fixed input-order prompt batches preserve the extraction convention, but do
    not promise bitwise agreement with historical parent-grouped GPU batches.
    """
    report = progress or (lambda message: print(f"[extract] {message}", flush=True))
    if not isinstance(batch_size, int) or batch_size < 1 or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("batch_size and chunk_size must be positive integers")
    device = str(device)
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("Extraction device must be cpu or cuda[:index]")
    dtype = dtype or ("float32" if device == "cpu" else "bfloat16")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16 or bfloat16")
    data, model_path = Path(data_dir), Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")
    prompt_path = data / "prompts.jsonl.gz" if data.is_dir() else data
    report("load and validate prompts")
    records = _records(prompt_path)
    output = Path(output_dir) if output_dir is not None else prompt_path.parent
    manifest_path = output / "features_manifest.json"
    existing = list(output.glob("features*.npz")) if output.exists() else []
    if not resume and (existing or manifest_path.exists()):
        raise FileExistsError("Feature cache already exists; use --resume for a matching checkpoint or choose an empty --output-dir")
    if resume and existing and not manifest_path.exists():
        raise ValueError("Existing feature files have no extraction checkpoint; choose a new output directory")
    if output.exists() and list(output.glob("features*.npz.partial")):
        raise ValueError("An incomplete feature file exists; choose a new output directory")
    report("verify local model and tokenizer identity")
    identity = _model_files(model_path, report)
    report(f"load local model ({device}, {dtype}); no download")
    tokenizer, model, torch = _load_model(model_path, device, dtype)
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "unavailable"
    configuration = {
        "prompt_count": len(records), "prompts_file": prompt_path.name,
        "prompts_sha256": _sha(prompt_path),
        "model_files": identity, "model_type": model.config.model_type,
        "hidden_size": int(model.config.hidden_size),
        "max_position_embeddings": int(model.config.max_position_embeddings),
        "device": device, "dtype": dtype, "attention_backend": "sdpa",
        "torch": str(torch.__version__), "transformers": transformers_version,
        "batch_size": batch_size, "chunk_size": chunk_size,
        "batch_order": "input JSONL order; consecutive prompt batches within chunks",
        "padding": "left", "position_ids": "attention_mask.cumsum(-1)-1; padding positions zero",
        "representation": "Qwen2Model.last_hidden_state after final norm, last real input token",
        "add_special_tokens": False, "truncation": False, "output_dtype": "float32",
        "prompt_rendering": "existing rendered_prompt verbatim; otherwise tokenizer user chat template with generation prompt",
        "rendered_prompt_count": sum(row["rendered_prompt"] is not None for row in records),
    }
    names = (["features.npz"] if len(records) <= chunk_size else
             [f"features_{i:03d}.npz" for i in range((len(records) + chunk_size - 1) // chunk_size)])
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = _validate_resume(manifest, configuration, output, records, names)
        if completed == len(records):
            manifest["status"] = "complete"
            manifest.pop("error", None)
            _write_manifest(manifest_path, manifest)
            report(f"validated complete cache: {completed}/{len(records)} prompts")
            return output
    else:
        manifest = {"schema_version": 1, "status": "running", "configuration": configuration,
                    "completed_rows": 0, "feature_dimension": None, "chunks": []}
        completed = 0
    manifest["status"] = "running"
    manifest.pop("error", None)
    _write_manifest(manifest_path, manifest)
    try:
        for chunk_index in range(len(manifest["chunks"]), len(names)):
            start, end = completed, min(completed + chunk_size, len(records))
            vectors = []
            for batch_start in range(start, end, batch_size):
                batch = records[batch_start:min(batch_start + batch_size, end)]
                matrix = _batch_vectors(batch, tokenizer, model, torch, device)
                if matrix.shape != (len(batch), configuration["hidden_size"]) or matrix.dtype != np.float32:
                    raise ValueError("Model returned an unexpected representation width or dtype")
                vectors.append(matrix)
                report(f"forward prompts {batch_start + len(batch)}/{len(records)}; chunk {chunk_index + 1}/{len(names)}")
            path = output / names[chunk_index]
            temporary = path.with_name(path.name + ".partial")
            report(f"write checkpoint {chunk_index + 1}/{len(names)}")
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, sample_id=np.asarray([row["sample_id"] for row in records[start:end]]),
                                    features=np.concatenate(vectors))
            # A hard link publishes the complete file without replacing any path.
            os.link(temporary, path)
            temporary.unlink()
            completed = end
            manifest["feature_dimension"] = configuration["hidden_size"]
            manifest["completed_rows"] = completed
            manifest["chunks"].append({"file": path.name, "start": start, "rows": end - start,
                                       "sha256": _sha(path)})
            _write_manifest(manifest_path, manifest)
        manifest["status"] = "complete"
        _write_manifest(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        _write_manifest(manifest_path, manifest)
        raise
    report(f"complete: {completed}/{len(records)} prompts; local float32 features saved")
    return output
