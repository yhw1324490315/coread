"""Extraction cache checks and optional real, offline tiny-Qwen2 integration."""

import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from consistency_readout import extraction


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".verification"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="extraction_test_", dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.data = self.work / "data"
        self.data.mkdir()
        self.model_path = self.work / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
        self.rows = [{"sample_id": f"id_{i}", "rendered_prompt": f"prompt {i}"} for i in range(7)]
        self.write_prompts(self.rows)
        self.backend = (None, SimpleNamespace(config=SimpleNamespace(
            model_type="qwen2", hidden_size=4, max_position_embeddings=128)),
            SimpleNamespace(__version__="test-fixture"))

    def write_prompts(self, rows):
        with gzip.open(self.data / "prompts.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def vectors(self, rows, *args):
        return np.array([[float(row["sample_id"].split("_")[-1]), 1., 2., 3.]
                         for row in rows], dtype=np.float32)

    def extract(self, output=None, **options):
        return extraction.extract_features(self.data, self.model_path,
                                           output_dir=output or self.work / "features",
                                           batch_size=2, chunk_size=3, device="cpu",
                                           progress=lambda message: None, **options)

    def patches(self, vector_callback=None):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(extraction, "_load_model", return_value=self.backend))
        stack.enter_context(patch.object(extraction, "_batch_vectors", side_effect=vector_callback or self.vectors))
        return stack

    def test_chunk_ids_full_dimension_and_manifest(self):
        with self.patches():
            output = self.extract()
        manifest = json.loads((output / "features_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed_rows"], 7)
        self.assertEqual(manifest["configuration"]["dtype"], "float32")
        self.assertEqual(manifest["configuration"]["rendered_prompt_count"], 7)
        self.assertEqual([row["rows"] for row in manifest["chunks"]], [3, 3, 1])
        seen = []
        for chunk in manifest["chunks"]:
            file = output / chunk["file"]
            self.assertEqual(extraction._sha(file), chunk["sha256"])
            with np.load(file, allow_pickle=False) as bundle:
                self.assertEqual(bundle["features"].dtype, np.float32)
                self.assertEqual(bundle["features"].shape[1], 4)
                seen.extend(bundle["sample_id"].tolist())
        self.assertEqual(seen, [row["sample_id"] for row in self.rows])

    def test_interrupted_run_resumes_only_verified_chunks(self):
        calls = 0

        def interrupted(rows, *args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated forward interruption")
            return self.vectors(rows)

        output = self.work / "features"
        with self.patches(interrupted), self.assertRaisesRegex(RuntimeError, "interruption"):
            self.extract(output)
        failed = json.loads((output / "features_manifest.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["completed_rows"], 3)
        first_chunk = (output / "features_000.npz").read_bytes()
        processed = []

        def resumed(rows, *args):
            processed.extend(row["sample_id"] for row in rows)
            return self.vectors(rows)

        with self.patches(resumed):
            self.extract(output, resume=True)
        self.assertEqual(processed, ["id_3", "id_4", "id_5", "id_6"])
        self.assertEqual((output / "features_000.npz").read_bytes(), first_chunk)
        self.assertEqual(json.loads((output / "features_manifest.json").read_text())["status"], "complete")

    def test_cache_rejects_overwrite_model_change_and_corruption(self):
        with self.patches():
            output = self.extract()
            with self.assertRaises(FileExistsError):
                self.extract(output)
            config = self.model_path / "config.json"
            config.write_text('{"model_type":"qwen2","modified":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                self.extract(output, resume=True)
            config.write_text('{"model_type":"qwen2"}', encoding="utf-8")
            with (output / "features_000.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                self.extract(output, resume=True)

    def test_complete_resume_never_repeats_forward_and_repairs_status(self):
        with self.patches():
            output = self.extract()
        path = output / "features_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["status"] = "running"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.patches(lambda *args: self.fail("Complete resume repeated model inference")):
            self.extract(output, resume=True)
        self.assertEqual(json.loads(path.read_text())["status"], "complete")

    def test_existing_uncheckpointed_features_are_preserved(self):
        output = self.work / "features"
        output.mkdir()
        file = output / "features.npz"
        np.savez_compressed(file, sample_id=np.array(["old"]), features=np.ones((1, 4), np.float32))
        before = file.read_bytes()
        with self.assertRaises(FileExistsError):
            self.extract(output)
        with self.assertRaisesRegex(ValueError, "no extraction checkpoint"):
            self.extract(output, resume=True)
        self.assertEqual(before, file.read_bytes())

    def test_bad_prompts_fail_before_model_loading(self):
        with patch.object(extraction, "_load_model") as load:
            self.write_prompts([self.rows[0], self.rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.extract()
            self.write_prompts([{"sample_id": "empty", "text": ""}])
            with self.assertRaisesRegex(ValueError, "Missing prompt"):
                self.extract()
            load.assert_not_called()


HAVE_EXTRACTION = all(importlib.util.find_spec(name) is not None for name in ["torch", "transformers", "tokenizers"])


@unittest.skipUnless(HAVE_EXTRACTION, "Install .[extract] to run the offline tiny-Qwen2 test")
class TinyQwenExtractionTests(unittest.TestCase):
    def test_real_model_post_norm_last_token_padding_and_chat_rendering(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        temporary_root = ROOT / ".verification"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tiny_qwen_extraction_", dir=temporary_root) as temporary:
            work = Path(temporary)
            local = work / "model"
            backend = Tokenizer(BPE(unk_token="[UNK]"))
            backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
            backend.train_from_iterator(["hello world words answer", "world answer hello"],
                                        BpeTrainer(vocab_size=285, initial_alphabet=ByteLevel.alphabet(),
                                                   special_tokens=["[PAD]", "[UNK]", "[EOS]"]))
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}{% if add_generation_prompt %} answer{% endif %}"
            tokenizer.padding_side = "left"
            tokenizer.save_pretrained(local)
            torch.manual_seed(19)
            config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                 max_position_embeddings=32, attention_dropout=0,
                                 pad_token_id=0, eos_token_id=2)
            reference = Qwen2ForCausalLM(config).eval()
            reference.save_pretrained(local)
            tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
            tokenizer.padding_side = "left"
            prompts = work / "new_prompts.jsonl"
            rows = [{"sample_id": "one", "rendered_prompt": "hello"},
                    {"sample_id": "three", "rendered_prompt": "hello world words"},
                    {"sample_id": "two", "rendered_prompt": "hello world"},
                    {"sample_id": "raw", "text": "world"}]
            prompts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            threads = torch.get_num_threads()
            torch.set_num_threads(2)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    one = extraction.extract_features(prompts, local, work / "one", batch_size=1,
                                                       chunk_size=10, device="cpu")
                    batch = extraction.extract_features(prompts, local, work / "batch", batch_size=4,
                                                         chunk_size=10, device="cpu")
                with np.load(one / "features.npz", allow_pickle=False) as z:
                    single_vectors = z["features"]
                with np.load(batch / "features.npz", allow_pickle=False) as z:
                    batch_vectors = z["features"]
                    self.assertEqual(z["sample_id"].tolist(), [row["sample_id"] for row in rows])
                texts = ["hello", "hello world words", "hello world", "world answer"]
                inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
                positions = (inputs["attention_mask"].cumsum(-1) - 1).masked_fill(inputs["attention_mask"] == 0, 0)
                captured = []
                handle = reference.model.norm.register_forward_hook(lambda module, arguments, output: captured.append(output.detach()))
                try:
                    with torch.inference_mode():
                        reference(**inputs, position_ids=positions, use_cache=False)
                finally:
                    handle.remove()
                expected = captured[0][:, -1].float().numpy()
                np.testing.assert_allclose(batch_vectors, expected, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(single_vectors, expected, rtol=2e-5, atol=2e-6)
                self.assertEqual(batch_vectors.dtype, np.float32)
                self.assertEqual(batch_vectors.shape, (4, 16))
                too_long = work / "too_long.jsonl"
                too_long.write_text(json.dumps(dict(sample_id="long", rendered_prompt=" ".join(["hello"] * 33))) + "\n", encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "context limit"):
                    extraction.extract_features(too_long, local, work / "too_long", device="cpu")
            finally:
                torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
