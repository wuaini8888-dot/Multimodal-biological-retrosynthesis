import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from torch.amp import autocast, GradScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from graph_utils import ATOM_FEAT_DIM
from ec_data import HierarchicalECDataset, graph_collate_fn
from ec_model import BioEC_EnzymePredictor, BioDualModalEncoder

PRETRAINED_PATH = "/data/stu1/ml_project/bioec_retro1/EC/deepchem_bert"
DATA_DIR = "/data/stu1/ml_project/bioec_retro1/dataset/EC/clean_processed_csv"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
SAVE_DIR = "save_ec_siamese_optimized"
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 100
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)

    print("Loading training set...")
    train_dataset = HierarchicalECDataset(TRAIN_PATH, tokenizer, max_len=512, is_train=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=graph_collate_fn, num_workers=8, pin_memory=True, prefetch_factor=2
    )

    label_maps_path = os.path.join(SAVE_DIR, 'label_maps.pkl')
    with open(label_maps_path, 'wb') as f:
        pickle.dump(train_dataset.label_maps, f)

    num_classes_list = [len(m) for m in train_dataset.label_maps]
    print(f"Number of classes per level: {num_classes_list}")

    print("Initializing Tri-Modal Neural-Symbolic System...")
    encoder = BioDualModalEncoder(PRETRAINED_PATH, atom_feat_dim=ATOM_FEAT_DIM)
    model = BioEC_EnzymePredictor(encoder, num_classes_list, hidden_dim=512).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler('cuda')

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in pbar:
            # 1. Extract all features of substrate R and move to GPU
            input_ids_R = batch['input_ids_R'].to(DEVICE, non_blocking=True)
            attention_mask_R = batch['attention_mask_R'].to(DEVICE, non_blocking=True)
            graph_x_R = batch['graph_x_R'].to(DEVICE, non_blocking=True)
            graph_adj_R = batch['graph_adj_R'].to(DEVICE, non_blocking=True)
            fp_indices_R = batch['fp_indices_R'].to(DEVICE, non_blocking=True)
            fp_mask_R = batch['fp_mask_R'].to(DEVICE, non_blocking=True)
            fp_dense_R = batch['fp_dense_R'].to(DEVICE, non_blocking=True)

            # 2. Extract all features of product P and move to GPU
            input_ids_P = batch['input_ids_P'].to(DEVICE, non_blocking=True)
            attention_mask_P = batch['attention_mask_P'].to(DEVICE, non_blocking=True)
            graph_x_P = batch['graph_x_P'].to(DEVICE, non_blocking=True)
            graph_adj_P = batch['graph_adj_P'].to(DEVICE, non_blocking=True)
            fp_indices_P = batch['fp_indices_P'].to(DEVICE, non_blocking=True)
            fp_mask_P = batch['fp_mask_P'].to(DEVICE, non_blocking=True)
            fp_dense_P = batch['fp_dense_P'].to(DEVICE, non_blocking=True)

            labels = batch['labels'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast('cuda'):
                l1, l2, l3, l4 = model(
                    input_ids_R, attention_mask_R, graph_x_R, graph_adj_R, fp_indices_R, fp_mask_R, fp_dense_R,
                    input_ids_P, attention_mask_P, graph_x_P, graph_adj_P, fp_indices_P, fp_mask_P, fp_dense_P,
                    targets=labels
                )

                loss = 0.2 * criterion(l1, labels[:, 0]) + \
                       0.2 * criterion(l2, labels[:, 1]) + \
                       0.2 * criterion(l3, labels[:, 2]) + \
                       0.4 * criterion(l4, labels[:, 3])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved -> {model_path}")

if __name__ == "__main__":
    main()