# Athena: Fallacy and Hate Speech Detector

Athena is a multi-task NLP model that reads a piece of text and answers three questions at once:

1. **Does it contain a logical fallacy, and which one?** (8 fallacy types, or `none`)
2. **Is it hateful?** (`normal` or `hateful`)
3. **Which words make it hateful?** (a per-word "rationale" label)

It uses one shared [ELECTRA](https://huggingface.co/google/electra-base-discriminator) encoder with three task-specific heads, trained jointly on a fallacy dataset and [HateXplain](https://github.com/hate-alert/HateXplain). This was my BSc Computer Science dissertation project at the University of Southampton.

## Results

Macro F1 on the held-out test sets:

| Task | Macro F1 |
| --- | --- |
| Fallacy classification (9 classes) | **0.83** |
| Hate speech detection (2 classes) | **0.77** |
| Rationale identification (per word) | **0.78** |

**What moved the numbers.** Fallacy F1 started at 0.67. Instead of tuning the model further, I went back to the data, found annotation errors in the fallacy dataset, corrected them, and F1 rose to 0.83. I also tried two common fixes for class imbalance and dropped both because they made results worse:

- Synonym-swap data augmentation lowered fallacy F1 to 0.80 and hate F1 to 0.71. The code is still in `data.py` behind `--augment`, so the result can be reproduced.
- LLM-generated synthetic training data also lowered F1.

## How it works

```
                 text
                  │
      ELECTRA-base encoder (shared)
          │                  │
   [CLS] vector         every token
      │        │             │
 fallacy head  hate head   rationale head
  (9 classes)  (2 classes)  (2 classes per token)
```

- Each head is `LayerNorm → Dropout → Linear`.
- Training mixes fallacy and hate examples in one dataset. Every row carries all three label columns, and the tasks a row doesn't belong to are set to `-100`, so they're ignored in the loss.
- Loss = 1.0 × fallacy + 1.0 × hate + 0.5 × rationale.
- The best checkpoint is chosen by the average of the three macro F1 scores on the validation set.

## Project structure

| File | What it does |
| --- | --- |
| `data.py` | Loads both datasets, builds train/dev/test splits, aligns word-level rationale labels to sub-word tokens, and (optionally) augments data |
| `model.py` | The multi-task ELECTRA model, plus saving and loading |
| `train.py` | Command-line training script: trains, evaluates on the test set, saves the model, confusion matrices and mispredictions |
| `evaluate.py` | Metrics (per-task and per-class F1), confusion matrices, misprediction export |
| `predict.py` | Classifies a single piece of text with a trained model |
| `tests/` | Automated tests that run offline in about 15 seconds |

## Setup

You need Python 3.11 or newer (tested on 3.11 and 3.13). A CUDA GPU is strongly recommended for training. On a GPU, training uses mixed precision (fp16) automatically. Prediction and the tests run fine on CPU.

```bash
git clone https://github.com/mmkktt2003/Athena-Fallacy-And-HateSpeech-Detector.git
cd Athena-Fallacy-And-HateSpeech-Detector

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

On a GPU machine, install the PyTorch build for your CUDA version first, following [pytorch.org](https://pytorch.org/get-started/locally/), then run the `pip install -r requirements.txt` line.

The first training run downloads `google/electra-base-discriminator` (about 440 MB) from Hugging Face.

## Data

Neither dataset is included in this repository. Put them anywhere and pass the paths to `train.py`. A `data/` folder is a tidy choice:

```bash
mkdir data
```

### Hate speech: HateXplain (public)

```bash
git clone https://github.com/hate-alert/HateXplain.git
cp HateXplain/Data/dataset.json data/dataset.json
```

How Athena reads it:

- **Label:** a post is `hateful` if **any** annotator marked it `hatespeech` or `offensive`. Otherwise it's `normal`.
- **Rationale:** a word counts as a rationale word if a majority of the annotators who highlighted that post highlighted that word.
- **Split:** random 80 / 10 / 10 train / dev / test, seed 42.

### Fallacies

A JSON list where each item has the text and its fallacy label. The label can be a string or a one-item list:

```json
[
  {"comment": "Everyone I know uses it, so it must be the best.", "fallacy": "appeal_to_majority"},
  {"comment": "I had a great time at the beach today.", "fallacy": ["none"]}
]
```

- **Classes:** the 8 fallacy types plus `none`. The label-to-index map is built from the labels found in the file, sorted alphabetically.
- **Split:** random 60 / 20 / 20 train / dev / test, seed 42.

## Train

```bash
python train.py \
    --fallacy-data data/combined_fallacy_data.json \
    --hate-data data/dataset.json \
    --output-dir results_electra_multitask
```

The defaults are the settings used for the reported results:

| Option | Default | Notes |
| --- | --- | --- |
| `--model-name` | `google/electra-base-discriminator` | Any ELECTRA checkpoint, or a local folder |
| `--max-len` | `128` | Kept at 128 to fit in GPU memory |
| `--epochs` | `3` | Early stopping with patience 3 |
| `--batch-size` | `4` | |
| `--gradient-accumulation-steps` | `8` | Effective batch size of 32 |
| `--learning-rate` | `2e-5` | Weight decay 0.01 |
| `--seed` | `42` | |
| `--augment` | off | Synonym augmentation. It lowered every score, so it's off for the final model |

What you get in `results_electra_multitask/`:

```
final_model/                                  trained model: load with predict.py
  config.json, electra_config.json, pytorch_model.bin,
  tokenizer files, fallacy_map.json, hate_map.json
test_results.json                             all test-set metrics, including per-class F1
fallacy_classification_confusion_matrix.png
hate_speech_classification_confusion_matrix.png
mispredictions/                               up to 50 wrong predictions per task, with confidence and logits
checkpoint-*/                                 best Trainer checkpoint
```

The script also prints a per-class F1 breakdown and a few mispredicted examples at the end.

## Predict

```bash
python predict.py \
    --model results_electra_multitask/final_model \
    --text "If we let them change this rule, soon there will be no rules at all."
```

Output:

```
Text:    If we let them change this rule, soon there will be no rules at all.
Fallacy: <predicted fallacy type or none>
Hate:    <normal or hateful>
```

A saved model loads entirely from its own folder, so prediction works offline once training is done.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

The tests build a tiny random ELECTRA model and small synthetic datasets on the fly. They need no downloads and no GPU, and take about 15 seconds. They cover:

- **Data loading:** label maps, split sizes, how hate labels are defined, majority-vote rationales.
- **Sub-word label alignment:** special tokens and padding are ignored.
- **The multi-task dataset:** each row only trains its own task.
- **The model:** output shapes, masked loss, gradients.
- **Save and load:** a saved model must load offline and give identical outputs.
- **Metrics:** perfect predictions must score F1 = 1.0.
- **The full pipeline:** `train.py` and then `predict.py`, run as real commands.

The suite passes on Python 3.13 with PyTorch 2.15 and transformers 5.24, and on Python 3.11 with PyTorch 2.3 and transformers 4.41. transformers 4.40 and older don't work, because `train.py` uses the newer `eval_strategy` argument.

## Limitations

- The hate label counts `offensive` as `hateful`, and needs only one annotator to say so. That makes the positive class large (about 74% of HateXplain posts) and differs from HateXplain's own majority-vote labels. So hate F1 here isn't directly comparable to the HateXplain paper.
- The splits are random, not HateXplain's official `post_id_divisions.json` splits.
- Inputs are split on whitespace before tokenising, so punctuation stays attached to words (`rules.`). Training and prediction do the same, so they stay consistent.
- `predict.py` shows the fallacy and hate predictions. The rationale head is trained and evaluated, but `predict.py` doesn't print the highlighted words yet.
