import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import pickle
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """Load data and ensure required columns are present"""
    logger.info(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
        # Ensure data contains the required columns
        if not all(col in data.columns for col in ["Products", "EC"]):
            raise ValueError(f"File {file_path} missing required columns ('Products' or 'EC')")
        data = data.dropna(subset=["Products", "EC"])
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise


def create_label_mapping(train_data):
    """Create label mapping based solely on training data"""
    train_labels = train_data["EC"].values.tolist()
    unique_labels = sorted(list(set(train_labels)))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    logger.info(f"Created label mapping with {len(unique_labels)} unique classes")
    return label_to_id, id_to_label


def filter_unknown_labels(data, label_to_id):
    """Filter out data that is not present in the label mapping"""
    original_count = len(data)
    data = data[data["EC"].isin(label_to_id.keys())]
    filtered_count = original_count - len(data)
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} samples with unknown labels")
    return data


def tokenize_data(tokenizer, smiles_list, labels, max_length=512):
    """Convert SMILES to model input features"""
    try:
        encodings = tokenizer(
            smiles_list,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise


def main():
    # Load data (keep original splits)
    train_data = load_data("/data/stu1/ml_project/A77/EC4/data/step3/train.csv")
    valid_data = load_data("/data/stu1/hjq_pycharm_project/A77/EC4/data/step3/valid.csv")
    test_data = load_data("/data/stu1/hjq_pycharm_project/A77/EC4/data/step3/test.csv")

    # Create label mapping based solely on training data
    label_to_id, id_to_label = create_label_mapping(train_data)

    # Filter unknown labels in validation and test sets
    valid_data = filter_unknown_labels(valid_data, label_to_id)
    test_data = filter_unknown_labels(test_data, label_to_id)

    # Prepare datasets
    train_smiles = train_data["Products"].values.tolist()
    train_labels = [label_to_id[label] for label in train_data["EC"].values.tolist()]

    valid_smiles = valid_data["Products"].values.tolist()
    valid_labels = [label_to_id[label] for label in valid_data["EC"].values.tolist()]

    test_smiles = test_data["Products"].values.tolist()
    test_labels = [label_to_id[label] for label in test_data["EC"].values.tolist()]

    # Shuffle training data (keep SMILES and labels aligned)
    combined = list(zip(train_smiles, train_labels))
    random.shuffle(combined)
    train_smiles, train_labels = zip(*combined)
    train_smiles = list(train_smiles)  # Ensure it is a list
    train_labels = list(train_labels)  # Ensure it is a list

    # Load tokenizer
    local_model_path = "/data/stu1/hjq_pycharm_project/A55/EC4/zinc250k_v2_40k/zinc250k_v2_40k"
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise

    # Calculate maximum length (based solely on the training set)
    max_len = max(len(tokenizer.encode(smiles, add_special_tokens=True)) for smiles in train_smiles)
    max_seq_length = min(512, max(300, max_len))
    logger.info(f"Max sequence length: {max_seq_length} (based on training set)")

    # Tokenize all datasets
    logger.info("Tokenizing data...")
    train_encodings = tokenize_data(tokenizer, train_smiles, train_labels, max_seq_length)
    valid_encodings = tokenize_data(tokenizer, valid_smiles, valid_labels, max_seq_length)
    test_encodings = tokenize_data(tokenizer, test_smiles, test_labels, max_seq_length)

    # Save results
    cache_dir = "/data/stu1/hjq_pycharm_project/A55/EC4/zinc250k_v2_40k/preprocessed"
    os.makedirs(cache_dir, exist_ok=True)

    # Save encoded data
    torch.save(train_encodings, os.path.join(cache_dir, "train_encodings.pt"))
    torch.save(valid_encodings, os.path.join(cache_dir, "valid_encodings.pt"))
    torch.save(test_encodings, os.path.join(cache_dir, "test_encodings.pt"))

    # Save label mapping
    with open(os.path.join(cache_dir, "label_map.pkl"), "wb") as f:
        pickle.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)

    # Save label mapping in text format
    with open(os.path.join(cache_dir, "label_map.txt"), "w") as f:
        for label, idx in label_to_id.items():
            f.write(f"{label}\t{idx}\n")

    logger.info("Preprocessing completed successfully")
    logger.info(
        f"Final dataset sizes - Train: {len(train_labels)}, Valid: {len(valid_labels)}, Test: {len(test_labels)}")


if __name__ == "__main__":
    main()