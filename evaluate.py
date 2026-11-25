"""Metrics, confusion matrices, and misprediction collection for LIME analysis."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(eval_pred):
    """Compute metrics for model evaluation with per-class breakdown"""
    logits, labels = eval_pred

    # Check the structure of labels
    if isinstance(labels, tuple):
        # If labels is a tuple, unpack it
        f_labels, h_labels, r_labels = labels
    else:
        # If labels is a dict, extract from it
        f_labels = labels.get('fallacy_labels')
        h_labels = labels.get('hate_labels')
        r_labels = labels.get('rationale_labels')

    # Handle different output formats for predictions
    if isinstance(logits, tuple):
        f_logits, h_logits, r_logits = logits
    else:
        f_logits = logits.get('fallacy_logits')
        h_logits = logits.get('hate_logits')
        r_logits = logits.get('rationale_logits')

    # Get predictions by taking argmax
    f_preds = np.argmax(f_logits, axis=-1)
    h_preds = np.argmax(h_logits, axis=-1)
    r_preds = np.argmax(r_logits.reshape(-1, 2), axis=-1).reshape(r_logits.shape[0], -1)

    # Create masks for valid labels
    valid_f = f_labels != -100
    valid_h = h_labels != -100
    valid_r = r_labels.reshape(-1) != -100

    metrics = {}

    # Fallacy metrics - with per-class breakdown
    if valid_f.any():
        p_f, r_f, f1_f, _ = precision_recall_fscore_support(
            f_labels[valid_f], f_preds[valid_f], average="macro"
        )
        metrics.update({
            "f1_fallacy": f1_f,
            "precision_fallacy": p_f,
            "recall_fallacy": r_f
        })

        _, _, f1_per_class, support_per_class = precision_recall_fscore_support(
            f_labels[valid_f], f_preds[valid_f], average=None
        )
        unique_classes = np.unique(f_labels[valid_f])
        for cls in unique_classes:
            idx = np.where(unique_classes == cls)[0][0]
            metrics[f"f1_fallacy_class_{cls}"] = f1_per_class[idx]
            metrics[f"support_fallacy_class_{cls}"] = int(support_per_class[idx])
    else:
        metrics["f1_fallacy"] = 0.0

    # Hate speech metrics - with per-class breakdown
    if valid_h.any():
        p_h, r_h, f1_h, _ = precision_recall_fscore_support(
            h_labels[valid_h], h_preds[valid_h], average="macro"
        )
        metrics.update({
            "f1_hate": f1_h,
            "precision_hate": p_h,
            "recall_hate": r_h
        })

        _, _, f1_per_class, support_per_class = precision_recall_fscore_support(
            h_labels[valid_h], h_preds[valid_h], average=None
        )
        unique_classes = np.unique(h_labels[valid_h])
        for cls in unique_classes:
            idx = np.where(unique_classes == cls)[0][0]
            metrics[f"f1_hate_class_{cls}"] = f1_per_class[idx]
            metrics[f"support_hate_class_{cls}"] = int(support_per_class[idx])
    else:
        metrics["f1_hate"] = 0.0

    # Rationale metrics - with per-class breakdown
    if valid_r.any():
        flat_r_preds = r_preds.reshape(-1)[valid_r]
        flat_r_labels = r_labels.reshape(-1)[valid_r]

        p_r, r_r, f1_r, _ = precision_recall_fscore_support(
            flat_r_labels, flat_r_preds, average="macro"
        )
        metrics.update({
            "f1_rationale": f1_r,
            "precision_rationale": p_r,
            "recall_rationale": r_r
        })

        _, _, f1_per_class, support_per_class = precision_recall_fscore_support(
            flat_r_labels, flat_r_preds, average=None
        )
        unique_classes = np.unique(flat_r_labels)
        for cls in unique_classes:
            idx = np.where(unique_classes == cls)[0][0]
            metrics[f"f1_rationale_class_{cls}"] = f1_per_class[idx]
            metrics[f"support_rationale_class_{cls}"] = int(support_per_class[idx])
    else:
        metrics["f1_rationale"] = 0.0

    # Overall F1 (average of all tasks)
    metrics["f1"] = (metrics["f1_fallacy"] + metrics["f1_hate"] + metrics["f1_rationale"]) / 3

    return metrics


def create_confusion_matrix(y_true, y_pred, class_names, title, output_dir="."):
    """Create and save a confusion matrix as a PNG in output_dir"""
    # Pass every class index so the matrix matches class_names even if a class is missing from the test split
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()


def collect_task_predictions(model, dataset, device, task):
    """Run the model over a dataset and return (labels, predictions) for one task.

    task is either 'fallacy' or 'hate'.
    """
    label_key = f"{task}_labels"
    logit_key = f"{task}_logits"

    model.eval()  # disable dropout so predictions are deterministic

    preds, labels = [], []
    for example in dataset:
        if example[label_key] == -100:
            continue
        inputs = {
            'input_ids': torch.tensor([example['input_ids']]).to(device),
            'attention_mask': torch.tensor([example['attention_mask']]).to(device),
            'token_type_ids': torch.tensor([example['token_type_ids']]).to(device)
        }
        with torch.no_grad():
            outputs = model(**inputs)
        preds.append(torch.argmax(outputs[logit_key], dim=1).item())
        labels.append(example[label_key])

    return labels, preds


def save_mispredictions_for_lime(model, dataset, tokenizer, device, output_path, fallacy_map, hate_map,
                                 max_samples=100):
    """Save mispredictions for later LIME analysis"""
    model.eval()

    fallacy_mispredictions = []
    hate_mispredictions = []

    fallacy_count = 0
    hate_count = 0

    for i, example in enumerate(dataset):
        # Only process examples where we have ground truth labels for evaluation
        process_fallacy = example['fallacy_labels'] != -100
        process_hate = example['hate_labels'] != -100

        if not (process_fallacy or process_hate):
            continue

        inputs = {
            'input_ids': torch.tensor([example['input_ids']]).to(device),
            'attention_mask': torch.tensor([example['attention_mask']]).to(device),
            'token_type_ids': torch.tensor([example['token_type_ids']]).to(device)
        }

        with torch.no_grad():
            outputs = model(**inputs)

        original_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)

        if process_fallacy and fallacy_count < max_samples:
            pred = torch.argmax(outputs['fallacy_logits'], dim=1).item()
            true_label = example['fallacy_labels']

            if pred != true_label:
                fallacy_map_reverse = {v: k for k, v in fallacy_map.items()}
                pred_label = fallacy_map_reverse.get(pred, f"Unknown-{pred}")
                true_label_name = fallacy_map_reverse.get(true_label, f"Unknown-{true_label}")
                confidence = torch.softmax(outputs['fallacy_logits'], dim=1)[0, pred].item()

                fallacy_mispredictions.append({
                    'text': original_text,
                    'prediction': pred_label,
                    'true_label': true_label_name,
                    'confidence': confidence,
                    'example_id': i,
                    'raw_logits': outputs['fallacy_logits'][0].cpu().numpy().tolist()
                })
                fallacy_count += 1

        if process_hate and hate_count < max_samples:
            pred = torch.argmax(outputs['hate_logits'], dim=1).item()
            true_label = example['hate_labels']

            if pred != true_label:
                hate_map_reverse = {v: k for k, v in hate_map.items()}
                pred_label = hate_map_reverse.get(pred, f"Unknown-{pred}")
                true_label_name = hate_map_reverse.get(true_label, f"Unknown-{true_label}")
                confidence = torch.softmax(outputs['hate_logits'], dim=1)[0, pred].item()

                hate_mispredictions.append({
                    'text': original_text,
                    'prediction': pred_label,
                    'true_label': true_label_name,
                    'confidence': confidence,
                    'example_id': i,
                    'raw_logits': outputs['hate_logits'][0].cpu().numpy().tolist()
                })
                hate_count += 1

        # Early stop if we've collected enough samples
        if fallacy_count >= max_samples and hate_count >= max_samples:
            break

    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "fallacy_mispredictions.json"), "w") as f:
        json.dump(fallacy_mispredictions, f, indent=2)

    with open(os.path.join(output_path, "hate_mispredictions.json"), "w") as f:
        json.dump(hate_mispredictions, f, indent=2)

    print(f"Saved {len(fallacy_mispredictions)} fallacy mispredictions and "
          f"{len(hate_mispredictions)} hate speech mispredictions")

    return fallacy_mispredictions, hate_mispredictions


def print_task_metrics(test_results, task, class_names):
    """Print the per-class F1 breakdown for one task."""
    print(f"\n{task.upper()} CLASSIFICATION METRICS:")
    print(f"Overall F1: {test_results.get(f'eval_f1_{task}', 0):.4f}")

    for key, value in test_results.items():
        prefix = f"eval_f1_{task}_class_"
        if key.startswith(prefix):
            cls_idx = int(key.split("_")[-1])
            class_name = class_names.get(cls_idx, f"Class {cls_idx}")
            support = test_results.get(f"eval_support_{task}_class_{cls_idx}", 0)
            print(f"  {class_name}: F1={value:.4f}, Support={support}")
