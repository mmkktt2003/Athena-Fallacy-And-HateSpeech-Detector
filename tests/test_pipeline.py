"""Tests for data loading, the multi-task model, metrics, and the full train -> predict pipeline."""

import json
import os
import subprocess
import sys

import numpy as np
import torch
from transformers import ElectraTokenizerFast

from conftest import ROOT
from data import align_labels, load_fallacy_data, load_hate_speech_data, prepare_multi_task_datasets
from evaluate import compute_metrics
from model import ElectraForMultiTaskClassification

MAX_LEN = 32


# ---------- data.py ----------

def test_fallacy_loader_splits_60_20_20_and_maps_labels(fallacy_json):
    train_t, train_y, dev_t, dev_y, test_t, test_y, label_map = load_fallacy_data(fallacy_json)
    assert (len(train_t), len(dev_t), len(test_t)) == (36, 12, 12)
    # label map is sorted alphabetically and covers both string and list labels
    assert label_map == {"appeal_to_authority": 0, "appeal_to_majority": 1, "none": 2, "slippery_slope": 3}
    assert set(train_y + dev_y + test_y) <= set(label_map.values())
    assert all(isinstance(t, list) for t in train_t)  # texts are pre-split into words


def test_hate_loader_labels_and_majority_rationales(hate_json):
    out = load_hate_speech_data(hate_json)
    train_t, train_y, train_r, dev_t, dev_y, dev_r, test_t, test_y, test_r, hate_map = out
    assert hate_map == {"normal": 0, "hateful": 1}
    assert len(train_t) + len(dev_t) + len(test_t) == 60
    raw = json.load(open(hate_json, encoding="utf-8"))
    by_tokens = {tuple(v["post_tokens"]): v for v in raw.values()}
    for tokens, y, r in zip(train_t, train_y, train_r):
        post = by_tokens[tuple(tokens)]
        # a post is "hateful" if ANY annotator chose something other than "normal"
        assert y == int(any(a["label"] != "normal" for a in post["annotators"]))
        assert len(r) == len(tokens)
        if post["rationales"]:
            n = len(post["rationales"])
            expected = [int(sum(rr[i] for rr in post["rationales"]) >= n // 2 + 1) for i in range(len(tokens))]
            assert r == expected
        else:
            assert r == [0] * len(tokens)


def test_align_labels_handles_subwords_specials_and_padding(tiny_model_dir):
    tok = ElectraTokenizerFast.from_pretrained(tiny_model_dir)
    words = [["the", "weather", "unknownword"]]
    enc = tok(words, is_split_into_words=True, padding="max_length", truncation=True, max_length=10)
    aligned = align_labels(enc, [[1, 0, 1]], max_len=10)[0]
    assert len(aligned) == 10
    wids = enc.word_ids(0)
    for w, label in zip(wids, aligned):
        assert label == (-100 if w is None else [1, 0, 1][w])


def test_multi_task_dataset_masks_other_tasks(tiny_model_dir, fallacy_json, hate_json):
    tok = ElectraTokenizerFast.from_pretrained(tiny_model_dir)
    f_data, h_data = load_fallacy_data(fallacy_json), load_hate_speech_data(hate_json)
    train_ds, val_ds, test_ds, _ = prepare_multi_task_datasets(f_data, h_data, tok, max_len=MAX_LEN)
    n_f = len(f_data[1])
    for i, row in enumerate(train_ds):
        assert len(row["input_ids"]) == len(row["rationale_labels"]) == MAX_LEN
        if i < n_f:   # fallacy rows: no hate label and no rationale labels
            assert row["hate_labels"] == -100 and set(row["rationale_labels"]) == {-100}
            assert row["fallacy_labels"] != -100
        else:         # hate rows: no fallacy label
            assert row["fallacy_labels"] == -100 and row["hate_labels"] in (0, 1)


# ---------- model.py ----------

def test_forward_shapes_and_masked_loss(tiny_model_dir):
    model = ElectraForMultiTaskClassification(tiny_model_dir, num_fallacy_labels=4, num_hate_labels=2)
    ids = torch.randint(5, 20, (3, MAX_LEN))
    mask = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=mask)
    assert out["fallacy_logits"].shape == (3, 4)
    assert out["hate_logits"].shape == (3, 2)
    assert out["rationale_logits"].shape == (3, MAX_LEN, 2)
    assert out["loss"] is None  # no labels -> no loss

    # one fallacy row, two hate rows: each task's loss only uses its own rows
    out = model(input_ids=ids, attention_mask=mask,
                fallacy_labels=torch.tensor([1, -100, -100]),
                hate_labels=torch.tensor([-100, 0, 1]),
                rationale_labels=torch.full((3, MAX_LEN), -100))
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0
    out["loss"].backward()  # gradients flow


def test_save_and_load_roundtrip_works_offline(tiny_model_dir, tmp_path):
    torch.manual_seed(0)
    model = ElectraForMultiTaskClassification(tiny_model_dir, num_fallacy_labels=4, num_hate_labels=2).eval()
    save_dir = tmp_path / "saved"
    model.save_pretrained(str(save_dir))
    assert (save_dir / "electra_config.json").exists()

    # Point the saved config at a base model that does not exist: loading must not need it
    cfg = json.loads((save_dir / "config.json").read_text())
    cfg["model_name"] = "does-not-exist/electra-base"
    (save_dir / "config.json").write_text(json.dumps(cfg))

    loaded = ElectraForMultiTaskClassification.from_pretrained(str(save_dir)).eval()
    ids = torch.randint(5, 20, (2, MAX_LEN))
    with torch.no_grad():
        a, b = model(input_ids=ids), loaded(input_ids=ids)
    for key in ("fallacy_logits", "hate_logits", "rationale_logits"):
        assert torch.allclose(a[key], b[key], atol=1e-6)


# ---------- evaluate.py ----------

def test_compute_metrics_perfect_predictions_score_one():
    # 2 fallacy rows, 2 hate rows, rationale labels only on the hate rows
    f_labels = np.array([0, 1, -100, -100])
    h_labels = np.array([-100, -100, 0, 1])
    r_labels = np.array([[-100, -100], [-100, -100], [0, 1], [1, 0]])
    f_logits = np.array([[5, 0], [0, 5], [0, 0], [0, 0]], dtype=float)
    h_logits = np.array([[0, 0], [0, 0], [5, 0], [0, 5]], dtype=float)
    r_logits = np.zeros((4, 2, 2))
    r_logits[2, 0, 0] = r_logits[2, 1, 1] = r_logits[3, 0, 1] = r_logits[3, 1, 0] = 5
    m = compute_metrics(((f_logits, h_logits, r_logits), (f_labels, h_labels, r_labels)))
    assert m["f1_fallacy"] == m["f1_hate"] == m["f1_rationale"] == m["f1"] == 1.0


# ---------- full pipeline ----------

def test_train_then_predict_end_to_end(tiny_model_dir, fallacy_json, hate_json, tmp_path):
    out_dir = tmp_path / "results"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    train = subprocess.run(
        [sys.executable, os.path.join(ROOT, "train.py"),
         "--fallacy-data", fallacy_json, "--hate-data", hate_json,
         "--output-dir", str(out_dir), "--model-name", tiny_model_dir,
         "--epochs", "1", "--batch-size", "8", "--gradient-accumulation-steps", "1", "--max-len", str(MAX_LEN)],
        capture_output=True, text=True, env=env, timeout=600)
    assert train.returncode == 0, train.stdout[-2000:] + train.stderr[-2000:]

    final = out_dir / "final_model"
    for name in ("config.json", "electra_config.json", "pytorch_model.bin",
                 "fallacy_map.json", "hate_map.json"):
        assert (final / name).exists(), name
    results = json.loads((out_dir / "test_results.json").read_text())
    assert 0.0 <= results["eval_f1"] <= 1.0
    assert (out_dir / "fallacy_classification_confusion_matrix.png").exists()
    assert (out_dir / "mispredictions" / "hate_mispredictions.json").exists()

    pred = subprocess.run(
        [sys.executable, os.path.join(ROOT, "predict.py"), "--model", str(final),
         "--text", "everyone uses it so it must be good", "--max-len", str(MAX_LEN)],
        capture_output=True, text=True, env=env, timeout=300)
    assert pred.returncode == 0, pred.stderr[-2000:]
    fallacy_names = json.loads((final / "fallacy_map.json").read_text())
    lines = dict(l.split(":", 1) for l in pred.stdout.strip().splitlines() if ":" in l)
    assert lines["Fallacy"].strip() in fallacy_names
    assert lines["Hate"].strip() in ("normal", "hateful")
