"""Shared test fixtures.

Everything here is built locally, so the tests run offline in a few seconds:
a tiny randomly initialised ELECTRA (2 layers, hidden size 32) and small synthetic
datasets in the same JSON formats the real fallacy and HateXplain files use.
"""

import json
import os
import random
import sys

import pytest
import torch
from transformers import ElectraConfig, ElectraModel, ElectraTokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WORDS = ("everyone uses it so it must be good if we allow this then soon everything "
         "will be banned my doctor says so you are wrong people like them should not "
         "be here i love this song what a great day the weather is nice").split()
FALLACIES = ["appeal_to_majority", "slippery_slope", "appeal_to_authority", "none"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A tiny random ELECTRA encoder plus tokenizer saved to disk."""
    d = tmp_path_factory.mktemp("tiny-electra")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS))
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = ElectraTokenizerFast.from_pretrained(str(d))
    tokenizer.save_pretrained(str(d))
    torch.manual_seed(0)
    config = ElectraConfig(vocab_size=len(vocab), embedding_size=32, hidden_size=32,
                           num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                           max_position_embeddings=128)
    ElectraModel(config).save_pretrained(str(d))
    return str(d)


def _sentence(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


@pytest.fixture(scope="session")
def fallacy_json(tmp_path_factory):
    """Fallacy data: a JSON list of {"comment": str, "fallacy": str or [str]}."""
    rng = random.Random(1)
    rows = []
    for i in range(60):
        label = FALLACIES[i % len(FALLACIES)]
        # mix both label formats the loader accepts: a plain string and a one-item list
        rows.append({"comment": _sentence(rng, rng.randint(5, 15)),
                     "fallacy": [label] if i % 2 else label})
    path = tmp_path_factory.mktemp("data") / "fallacy.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="session")
def hate_json(tmp_path_factory):
    """Hate speech data in HateXplain's dataset.json format (dict keyed by post id)."""
    rng = random.Random(2)
    data = {}
    for i in range(60):
        tokens = _sentence(rng, rng.randint(5, 15)).split()
        labels = rng.choice([["normal"] * 3, ["normal", "normal", "offensive"],
                             ["hatespeech", "hatespeech", "offensive"]])
        hateful = any(l != "normal" for l in labels)
        rationales = ([[rng.randint(0, 1) for _ in tokens] for _ in range(2)] if hateful else [])
        data[f"post_{i}"] = {
            "post_id": f"post_{i}",
            "annotators": [{"label": l, "annotator_id": j, "target": ["None"]} for j, l in enumerate(labels)],
            "rationales": rationales,
            "post_tokens": tokens,
        }
    path = tmp_path_factory.mktemp("data") / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def run_in_tmp(tmp_path, monkeypatch):
    """data.py writes label-map JSON files to the working directory, so keep that out of the repo."""
    monkeypatch.chdir(tmp_path)
