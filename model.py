"""Multi-task ELECTRA model with shared encoder and three task-specific heads."""

import json
import os

import torch
from torch import nn
from transformers import ElectraConfig, ElectraModel

ENCODER_CONFIG_FILE = "electra_config.json"


class TaskSpecificHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_rate=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        # Apply layer normalization first
        normalized = self.layernorm(hidden_states)
        # Apply dropout
        dropped = self.dropout(normalized)
        # Apply classification layer
        logits = self.classifier(dropped)
        return logits


class ElectraForMultiTaskClassification(nn.Module):
    def __init__(self, model_name, num_fallacy_labels, num_hate_labels, task_weights=None, dropout_rate=0.1,
                 encoder_config=None):
        super().__init__()
        if encoder_config is not None:
            # Build the encoder architecture only; weights come from the saved state dict.
            # This avoids re-downloading the pretrained base model when loading a trained model.
            self.electra = ElectraModel(encoder_config)
        else:
            self.electra = ElectraModel.from_pretrained(model_name)
        hidden_size = self.electra.config.hidden_size

        # Use task-specific heads instead of simple layer norm + linear layers
        self.fallacy_head = TaskSpecificHead(hidden_size, num_fallacy_labels, dropout_rate)
        self.hate_head = TaskSpecificHead(hidden_size, num_hate_labels, dropout_rate)
        self.rationale_head = TaskSpecificHead(hidden_size, 2, dropout_rate)

        # Shared dropout for initial representation
        self.dropout = nn.Dropout(dropout_rate)

        # Default weights if not provided
        self.task_weights = task_weights or {
            'fallacy': 1.0,
            'hate': 1.0,
            'rationale': 0.5
        }

        self.loss_fct = nn.CrossEntropyLoss()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        self.electra.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self.electra.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                fallacy_labels=None, hate_labels=None, rationale_labels=None,
                return_dict=True, **kwargs):
        # Get the hidden states from the electra encoder
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # ELECTRA doesn't have a pooler, so we use the [CLS] token representation
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Get [CLS] token representation

        # Apply shared dropout first
        cls_output = self.dropout(cls_output)

        # Use task-specific heads for each task
        fallacy_logits = self.fallacy_head(cls_output)
        hate_logits = self.hate_head(cls_output)

        # For token classification (rationales), apply the head to every token.
        # Apply the dropout to the sequence first
        sequence_output = self.dropout(sequence_output)

        batch_size, seq_length, hidden_size = sequence_output.shape

        # All token positions are processed in a single matrix operation.
        # (An earlier version looped over seq_length and applied the head one
        # position at a time. Mathematically identical, but much slower.)
        flat_sequence = sequence_output.reshape(-1, hidden_size)
        flat_rationale_logits = self.rationale_head(flat_sequence)
        rationale_logits = flat_rationale_logits.view(batch_size, seq_length, 2)

        loss = None
        if fallacy_labels is not None and hate_labels is not None and rationale_labels is not None:
            # Create masks for valid labels (ignore -100)
            fallacy_mask = fallacy_labels != -100
            hate_mask = hate_labels != -100
            rationale_mask = rationale_labels.view(-1) != -100

            # Calculate losses only for valid labels
            fallacy_loss = (self.loss_fct(fallacy_logits[fallacy_mask], fallacy_labels[fallacy_mask])
                            if fallacy_mask.any() else torch.tensor(0.0).to(fallacy_logits.device))

            hate_loss = (self.loss_fct(hate_logits[hate_mask], hate_labels[hate_mask])
                         if hate_mask.any() else torch.tensor(0.0).to(hate_logits.device))

            # Flatten and reshape for token classification
            rationale_loss = (self.loss_fct(
                rationale_logits.view(-1, 2)[rationale_mask],
                rationale_labels.view(-1)[rationale_mask]
            ) if rationale_mask.any() else torch.tensor(0.0).to(rationale_logits.device))

            # Weighted sum of losses
            loss = (
                    self.task_weights['fallacy'] * fallacy_loss +
                    self.task_weights['hate'] * hate_loss +
                    self.task_weights['rationale'] * rationale_loss
            )

        if not return_dict:
            return (loss, fallacy_logits, hate_logits, rationale_logits)

        return {
            'loss': loss,
            'fallacy_logits': fallacy_logits,
            'hate_logits': hate_logits,
            'rationale_logits': rationale_logits
        }

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v.contiguous() for k, v in sd.items()}

    def save_pretrained(self, save_directory):
        """Save model weights and configuration"""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save configuration
        config = {
            "model_name": self.electra.config._name_or_path,
            "num_fallacy_labels": self.fallacy_head.classifier.out_features,
            "num_hate_labels": self.hate_head.classifier.out_features,
            "task_weights": self.task_weights
        }
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        # Save the encoder architecture so the model can be rebuilt offline
        self.electra.config.to_json_file(os.path.join(save_directory, ENCODER_CONFIG_FILE))

    @classmethod
    def from_pretrained(cls, load_directory):
        """Load model from saved weights and configuration"""
        try:
            with open(os.path.join(load_directory, "config.json"), "r", encoding="utf-8") as f:
                config = json.load(f)

            # Create model with saved configuration
            task_weights = config.get("task_weights", None)

            # Models saved with this version include the encoder config, so no download is needed.
            # Older saved models fall back to downloading the base model named in config.json.
            encoder_config = None
            encoder_config_path = os.path.join(load_directory, ENCODER_CONFIG_FILE)
            if os.path.exists(encoder_config_path):
                encoder_config = ElectraConfig.from_json_file(encoder_config_path)

            model = cls(
                config["model_name"],
                config["num_fallacy_labels"],
                config["num_hate_labels"],
                task_weights,
                encoder_config=encoder_config
            )

            # Load weights
            state_dict = torch.load(
                os.path.join(load_directory, "pytorch_model.bin"),
                map_location="cpu"
            )
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
