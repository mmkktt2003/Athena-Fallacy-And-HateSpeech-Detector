"""Train the multi-task fallacy / hate speech / rationale model.

Example:
    python train.py --fallacy-data data/combined_fallacy_data.json \
                    --hate-data data/dataset.json \
                    --output-dir ./results_electra_multitask
"""

import argparse
import gc
import json
import os
import random

import numpy as np
import torch
from transformers import (ElectraTokenizerFast, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from data import load_fallacy_data, load_hate_speech_data, prepare_multi_task_datasets
from evaluate import (collect_task_predictions, compute_metrics, create_confusion_matrix,
                      print_task_metrics, save_mispredictions_for_lime)
from model import ElectraForMultiTaskClassification


def parse_args():
    p = argparse.ArgumentParser(description="Train the Athena multi-task model")
    p.add_argument('--fallacy-data', required=True,
                   help='Path to the combined fallacy dataset JSON')
    p.add_argument('--hate-data', required=True,
                   help='Path to the HateXplain dataset JSON')
    p.add_argument('--output-dir', default='./results_electra_multitask')
    p.add_argument('--model-name', default='google/electra-base-discriminator')
    p.add_argument('--max-len', type=int, default=128,
                   help='Max sequence length (128 was used to fit in GPU memory)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--gradient-accumulation-steps', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--augment', action='store_true',
                   help='Enable synonym-based augmentation. NOT used in the final '
                        'model: it reduced F1 on every task.')
    return p.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    print("Initializing ELECTRA tokenizer...")
    tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name)

    print("Loading fallacy data...")
    f_data = load_fallacy_data(args.fallacy_data)

    print("Loading hate speech data...")
    h_data = load_hate_speech_data(args.hate_data)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Preparing multi-task datasets...")
    train_ds, val_ds, test_ds, label_maps = prepare_multi_task_datasets(
        f_data, h_data, tokenizer, max_len=args.max_len, augment=args.augment
    )
    fallacy_map, hate_map = label_maps

    print("Initializing multi-task model...")
    task_weights = {'fallacy': 1.0, 'hate': 1.0, 'rationale': 0.5}
    model = ElectraForMultiTaskClassification(
        model_name=args.model_name,
        num_fallacy_labels=len(fallacy_map),
        num_hate_labels=len(hate_map),
        task_weights=task_weights
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        logging_steps=100,
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Save label maps alongside the model so inference can decode predictions
    with open(os.path.join(final_model_dir, "fallacy_map.json"), "w") as f:
        json.dump(fallacy_map, f, indent=4)
    with open(os.path.join(final_model_dir, "hate_map.json"), "w") as f:
        json.dump(hate_map, f, indent=4)

    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)

    print("\n===== DETAILED EVALUATION RESULTS =====")
    fallacy_names = {v: k for k, v in fallacy_map.items()}
    hate_names = {v: k for k, v in hate_map.items()}
    rationale_names = {0: "Not-Rationale", 1: "Rationale"}

    print_task_metrics(test_results, "fallacy", fallacy_names)
    print_task_metrics(test_results, "hate", hate_names)
    print_task_metrics(test_results, "rationale", rationale_names)

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    print("\nGenerating confusion matrices...")
    f_labels, f_preds = collect_task_predictions(model, test_ds, device, "fallacy")
    fallacy_names_list = [""] * len(fallacy_map)
    for name, idx in fallacy_map.items():
        fallacy_names_list[idx] = name
    create_confusion_matrix(f_labels, f_preds, fallacy_names_list,
                            "Fallacy Classification Confusion Matrix", args.output_dir)

    h_labels, h_preds = collect_task_predictions(model, test_ds, device, "hate")
    hate_names_list = [""] * len(hate_map)
    for name, idx in hate_map.items():
        hate_names_list[idx] = name
    create_confusion_matrix(h_labels, h_preds, hate_names_list,
                            "Hate Speech Classification Confusion Matrix", args.output_dir)

    print(f"Confusion matrices saved to {args.output_dir}")

    print("Collecting mispredictions for LIME analysis...")
    mispredictions_path = os.path.join(args.output_dir, "mispredictions")
    fallacy_misp, hate_misp = save_mispredictions_for_lime(
        model, test_ds, tokenizer, device, mispredictions_path,
        fallacy_map, hate_map, max_samples=50
    )

    print("\nFallacy misprediction examples (first 3):")
    for i, ex in enumerate(fallacy_misp[:3]):
        print(f"Example {i + 1}:")
        print(f"  Text: {ex['text'][:100]}...")
        print(f"  Predicted: {ex['prediction']} (confidence: {ex['confidence']:.4f})")
        print(f"  True label: {ex['true_label']}")

    print("\nHate speech misprediction examples (first 3):")
    for i, ex in enumerate(hate_misp[:3]):
        print(f"Example {i + 1}:")
        print(f"  Text: {ex['text'][:100]}...")
        print(f"  Predicted: {ex['prediction']} (confidence: {ex['confidence']:.4f})")
        print(f"  True label: {ex['true_label']}")

    print("Training and evaluation complete!")


if __name__ == "__main__":
    main()
