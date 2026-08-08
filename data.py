"""Dataset loading, preprocessing, augmentation and multi-task dataset assembly."""

import json
import random
from collections import Counter

import nltk
from datasets import Dataset
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split


def make_fallacy_dataset(data, existing_label_map=None):
    """Process fallacy data and create label mapping"""
    texts = [e['comment'].split() for e in data]  # Split into words for tokenization

    # Handle single label or list of labels
    labels = [(e['fallacy'][0] if isinstance(e['fallacy'], list) else e['fallacy']) for e in data]

    # Create or use existing label map
    if existing_label_map is None:
        label_map = {l: i for i, l in enumerate(sorted(set(labels)))}
        with open("fallacy_label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=4)
    else:
        label_map = existing_label_map

    # Convert labels to indices, handle unknown labels
    encoded_labels = []
    for l in labels:
        if l in label_map:
            encoded_labels.append(label_map[l])
        else:
            print(f"Warning: Found unknown label '{l}', assigning to class 0")
            encoded_labels.append(0)

    return texts, encoded_labels, label_map


def load_fallacy_data(fallacy_path):
    """Load fallacy datasets with error handling"""
    try:
        with open(fallacy_path, encoding='utf-8') as f:
            fallacy_data = json.load(f)

        # Create label map from training data
        fallacy_texts, fallacy_labels, label_map = make_fallacy_dataset(fallacy_data)

        print(f"Fallacy data loaded: {len(fallacy_texts)} all")
        print(f"Fallacy labels: {label_map}")
        fallacy_train_texts, fallacy_test_dev_texts, fallacy_train_labels, fallacy_test_dev_labels = train_test_split(
            fallacy_texts, fallacy_labels, test_size=0.4, random_state=42)
        fallacy_test_texts, fallacy_dev_texts, fallacy_test_labels, fallacy_dev_labels = train_test_split(
            fallacy_test_dev_texts, fallacy_test_dev_labels, test_size=0.5, random_state=42)

        return (fallacy_train_texts, fallacy_train_labels, fallacy_dev_texts, fallacy_dev_labels,
                fallacy_test_texts, fallacy_test_labels, label_map)
    except Exception as e:
        print(f"Error loading fallacy data: {e}")
        raise


def load_hate_speech_data(path_or_data):
    """Load hate speech data from file or passed data"""
    hate_map = {"normal": 0, "hateful": 1}
    with open("hate_label_map.json", "w", encoding="utf-8") as fout:
        json.dump(hate_map, fout, indent=4)

    try:
        # Handle either path or pre-loaded data
        if isinstance(path_or_data, str):
            with open(path_or_data, encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = path_or_data

        # Convert dict to list if needed
        if isinstance(data, dict):
            data = list(data.values())

        texts, y, rationale_multi = [], [], []
        for instance in data:
            tokens = instance["post_tokens"]
            texts.append(tokens)

            # Get labels from annotators
            anns = [ann['label'].lower() for ann in instance.get('annotators', [])]
            y.append(1 if any(l != "normal" for l in anns) else 0)

            # Process rationales (annotations of which tokens are hateful)
            rats = instance.get('rationales', [])
            if rats and isinstance(rats[0], list):
                n = len(rats)
                merged = []
                threshold = n // 2 + 1  # Majority vote
                for i in range(len(tokens)):
                    count = sum(r[i] for r in rats)
                    merged.append(1 if count >= threshold else 0)
            else:
                merged = [0] * len(tokens)
            rationale_multi.append(merged)

        # Split into train/dev/test
        splits = train_test_split(texts, y, rationale_multi, test_size=0.2, random_state=42)
        train_t, temp_t, train_y, temp_y, train_r, temp_r = splits
        dev_t, test_t, dev_y, test_y, dev_r, test_r = train_test_split(
            temp_t, temp_y, temp_r, test_size=0.5, random_state=42)

        print(f"Hate speech data loaded: {len(train_t)} train, {len(dev_t)} dev, {len(test_t)} test")
        print(f"Hate speech class distribution - Train: {sum(train_y)}/{len(train_y)}, "
              f"Dev: {sum(dev_y)}/{len(dev_y)}, Test: {sum(test_y)}/{len(test_t)}")

        return (train_t, train_y, train_r, dev_t, dev_y, dev_r, test_t, test_y, test_r, hate_map)
    except Exception as e:
        print(f"Error loading hate speech data: {e}")
        raise


def align_labels(enc, raw_r, max_len=256):
    """Align word-level rationale labels with token-level inputs"""
    aligned = []
    for i in range(len(raw_r)):
        try:
            wids = enc.word_ids(batch_index=i)

            # Handle cases where word_ids might be None
            if wids is None:
                aligned.append([-100] * max_len)
                continue

            aligned_labels = []
            for w in wids:
                if w is None or w >= len(raw_r[i]):
                    # Special tokens or out of bounds
                    aligned_labels.append(-100)
                else:
                    # Get the corresponding label
                    aligned_labels.append(raw_r[i][w])

            # Ensure fixed length
            if len(aligned_labels) < max_len:
                aligned_labels.extend([-100] * (max_len - len(aligned_labels)))
            elif len(aligned_labels) > max_len:
                aligned_labels = aligned_labels[:max_len]

            aligned.append(aligned_labels)
        except Exception as e:
            print(f"Error aligning labels for item {i}: {e}")
            # Add placeholder in case of error
            aligned.append([-100] * max_len)

    return aligned


def augment_minority_classes(texts, labels, rationales=None, max_ratio=3.0):
    """Augment minority classes using synonym replacement and other techniques

    NOTE: this was evaluated and found to REDUCE performance (fallacy F1 0.83 -> 0.80,
    hate F1 0.77 -> 0.71). The final model is trained with augment=False. Kept here
    because the negative result is part of the project.
    """
    # Download wordnet if not already available
    try:
        wordnet.synsets('test')
    except LookupError:
        nltk.download('wordnet')

    class_counts = Counter(labels)

    if len(class_counts) <= 1:
        return texts, labels, rationales  # No need for augmentation

    # Initialize rationales if None
    if rationales is None:
        rationales = [None] * len(texts)

    # Get majority and minority class counts
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    # Don't augment too aggressively
    target_count = min(max_count, int(min_count * max_ratio))

    augmented_texts = texts.copy()
    augmented_labels = labels.copy()
    augmented_rationales = rationales.copy()

    def get_synonyms(word):
        """Get synonyms for a word using WordNet"""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        return synonyms

    def synonym_replacement(tokens, n_replacements=2):
        """Replace n words in the text with their synonyms"""
        # Create a copy of tokens to modify
        new_tokens = tokens.copy()

        # Only consider content words (exclude stopwords and short words)
        candidate_indices = [i for i, word in enumerate(tokens)
                             if len(word) > 3]  # Simple heuristic for content words

        # Skip if not enough candidates
        if not candidate_indices or len(candidate_indices) < n_replacements:
            return new_tokens

        # Randomly select indices to replace
        indices_to_replace = random.sample(candidate_indices,
                                           min(n_replacements, len(candidate_indices)))

        for idx in indices_to_replace:
            word = tokens[idx]
            synonyms = get_synonyms(word)
            if synonyms:
                # Replace with a random synonym
                new_tokens[idx] = random.choice(synonyms)

        return new_tokens

    def random_swap(tokens, n_swaps=1):
        """Randomly swap n pairs of words in the text"""
        new_tokens = tokens.copy()

        # Skip if text is too short
        if len(new_tokens) < 2:
            return new_tokens

        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(len(new_tokens)), 2)
            new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]

        return new_tokens

    def random_deletion(tokens, p=0.1):
        """Randomly delete words from the text with probability p"""
        # Don't delete too many words from short texts
        if len(tokens) < 5:
            return tokens

        new_tokens = []
        for token in tokens:
            if random.random() > p:
                new_tokens.append(token)

        # Make sure we don't delete everything
        if not new_tokens:
            # Keep at least one random token
            new_tokens = [random.choice(tokens)]

        return new_tokens

    # Augment minority classes
    for cls, count in class_counts.items():
        if count < target_count:
            # Select all instances of this class
            indices = [i for i, l in enumerate(labels) if l == cls]

            # How many more do we need?
            needed = target_count - count

            # Create new augmented instances
            for _ in range(needed):
                idx = random.choice(indices)
                original_tokens = texts[idx]

                # Apply different augmentation techniques with equal probability
                augmentation_choice = random.random()

                if augmentation_choice < 0.4:  # 40% synonym replacement
                    new_tokens = synonym_replacement(original_tokens)
                elif augmentation_choice < 0.7:  # 30% random swap
                    new_tokens = random_swap(original_tokens)
                else:  # 30% random deletion
                    new_tokens = random_deletion(original_tokens)

                # Add the augmented example
                augmented_texts.append(new_tokens)
                augmented_labels.append(labels[idx])

                # Handle rationales for augmented examples
                if rationales[idx] is not None:
                    # For synonym replacement and random swap, we can mostly keep the rationales
                    # For deletion, we need to adjust rationales
                    if augmentation_choice >= 0.7:  # deletion case
                        # Simplified: in practice you would track which tokens were deleted
                        augmented_rationales.append([0] * len(new_tokens))
                    else:
                        # For other techniques, copy the original rationales
                        # (This is an approximation; ideally you'd track transformed tokens)
                        if len(new_tokens) == len(original_tokens):
                            augmented_rationales.append(rationales[idx])
                        else:
                            # If lengths differ, create a placeholder rationale
                            augmented_rationales.append([0] * len(new_tokens))
                else:
                    augmented_rationales.append(None)

    print(f"Data augmentation: {len(texts)} -> {len(augmented_texts)} instances")
    return augmented_texts, augmented_labels, augmented_rationales


def prepare_multi_task_datasets(f_data, h_data, tokenizer, max_len=256, augment=False):
    """Prepare datasets for multi-task learning"""
    # Unpack data
    f_train_t, f_train_y, f_dev_t, f_dev_y, f_test_t, f_test_y, f_map = f_data
    h_train_t, h_train_y, h_train_r, h_dev_t, h_dev_y, h_dev_r, h_test_t, h_test_y, h_test_r, h_map = h_data

    # Optional data augmentation to handle class imbalance
    if augment:
        f_train_t, f_train_y, _ = augment_minority_classes(f_train_t, f_train_y)
        h_train_t, h_train_y, h_train_r = augment_minority_classes(h_train_t, h_train_y, h_train_r)

    # Tokenize inputs
    f_enc_train = tokenizer(f_train_t, is_split_into_words=True,
                            padding='max_length', truncation=True, max_length=max_len)
    f_enc_dev = tokenizer(f_dev_t, is_split_into_words=True,
                          padding='max_length', truncation=True, max_length=max_len)
    h_enc_train = tokenizer(h_train_t, is_split_into_words=True,
                            padding='max_length', truncation=True, max_length=max_len)
    h_enc_dev = tokenizer(h_dev_t, is_split_into_words=True,
                          padding='max_length', truncation=True, max_length=max_len)

    # Create placeholder rationales for fallacy data (not used)
    f_rat_train = [[-100] * max_len for _ in f_train_y]
    f_rat_dev = [[-100] * max_len for _ in f_dev_y]

    # Align rationale labels with tokens
    h_rat_train = align_labels(h_enc_train, h_train_r, max_len)
    h_rat_dev = align_labels(h_enc_dev, h_dev_r, max_len)

    # Create combined datasets with task-specific labels (-100 means "ignore for this task")
    train_ds = Dataset.from_dict({
        'input_ids': f_enc_train['input_ids'] + h_enc_train['input_ids'],
        'attention_mask': f_enc_train['attention_mask'] + h_enc_train['attention_mask'],
        'token_type_ids': f_enc_train['token_type_ids'] + h_enc_train['token_type_ids'],  # electra specific
        'fallacy_labels': f_train_y + [-100] * len(h_train_y),
        'hate_labels': [-100] * len(f_train_y) + h_train_y,
        'rationale_labels': f_rat_train + h_rat_train
    })

    val_ds = Dataset.from_dict({
        'input_ids': f_enc_dev['input_ids'] + h_enc_dev['input_ids'],
        'attention_mask': f_enc_dev['attention_mask'] + h_enc_dev['attention_mask'],
        'token_type_ids': f_enc_dev['token_type_ids'] + h_enc_dev['token_type_ids'],  # electra specific
        'fallacy_labels': f_dev_y + [-100] * len(h_dev_y),
        'hate_labels': [-100] * len(f_dev_y) + h_dev_y,
        'rationale_labels': f_rat_dev + h_rat_dev
    })

    # Also prepare test dataset for final evaluation
    f_enc_test = tokenizer(f_test_t, is_split_into_words=True,
                           padding='max_length', truncation=True, max_length=max_len)
    h_enc_test = tokenizer(h_test_t, is_split_into_words=True,
                           padding='max_length', truncation=True, max_length=max_len)

    f_rat_test = [[-100] * max_len for _ in f_test_y]
    h_rat_test = align_labels(h_enc_test, h_test_r, max_len)

    test_ds = Dataset.from_dict({
        'input_ids': f_enc_test['input_ids'] + h_enc_test['input_ids'],
        'attention_mask': f_enc_test['attention_mask'] + h_enc_test['attention_mask'],
        'token_type_ids': f_enc_test['token_type_ids'] + h_enc_test['token_type_ids'],  # electra specific
        'fallacy_labels': f_test_y + [-100] * len(h_test_y),
        'hate_labels': [-100] * len(f_test_y) + h_test_y,
        'rationale_labels': f_rat_test + h_rat_test
    })

    return train_ds, val_ds, test_ds, (f_map, h_map)
