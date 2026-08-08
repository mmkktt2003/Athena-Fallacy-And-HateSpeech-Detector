"""Run a trained Athena model on a single piece of text.

Example:
    python predict.py --model results_electra_multitask/final_model \
                      --text "Almost everyone I know uses Brand X, so it must be the best."
"""

import argparse
import json
import os

import torch
from transformers import ElectraTokenizerFast

from model import ElectraForMultiTaskClassification


def load_label_maps(model_dir):
    with open(os.path.join(model_dir, "fallacy_map.json"), encoding="utf-8") as f:
        fallacy_map = json.load(f)
    with open(os.path.join(model_dir, "hate_map.json"), encoding="utf-8") as f:
        hate_map = json.load(f)
    # Reverse them so we can turn predicted indices back into names
    return ({v: k for k, v in fallacy_map.items()},
            {v: k for k, v in hate_map.items()})


def predict(text, model, tokenizer, device, fallacy_names, hate_names, max_len=128):
    tokens = text.split()
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    fallacy_pred = torch.argmax(outputs['fallacy_logits'], dim=1).item()
    hate_pred = torch.argmax(outputs['hate_logits'], dim=1).item()

    return {
        'fallacy': fallacy_names.get(fallacy_pred, f"Unknown ({fallacy_pred})"),
        'hate': hate_names.get(hate_pred, f"Unknown ({hate_pred})")
    }


def main():
    p = argparse.ArgumentParser(description="Classify text with a trained Athena model")
    p.add_argument('--model', required=True, help='Path to the saved model directory')
    p.add_argument('--text', required=True, help='Text to classify')
    p.add_argument('--max-len', type=int, default=128)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = ElectraTokenizerFast.from_pretrained(args.model)
    model = ElectraForMultiTaskClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    fallacy_names, hate_names = load_label_maps(args.model)

    result = predict(args.text, model, tokenizer, device,
                     fallacy_names, hate_names, args.max_len)

    print(f"Text:    {args.text}")
    print(f"Fallacy: {result['fallacy']}")
    print(f"Hate:    {result['hate']}")


if __name__ == "__main__":
    main()
