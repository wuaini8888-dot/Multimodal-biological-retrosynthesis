import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ec_data import HierarchicalECDataset, graph_collate_fn
from ec_model import BioEC_EnzymePredictor, BioDualModalEncoder
from graph_utils import ATOM_FEAT_DIM

# Configuration parameters
PRETRAINED_PATH = "/data/stu1/ml_project/bioec_retro1/EC/deepchem_bert"
DATA_DIR = "/data/stu1/ml_project/bioec_retro1/dataset/EC/clean_processed_csv"
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAVE_DIR = "save_ec_siamese_optimized"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")
LABEL_MAP_PATH = os.path.join(SAVE_DIR, "label_maps.pkl")
BATCH_SIZE = 16
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'

def evaluate(model, test_loader, device):
    model.eval()

    correct_l1, correct_l2, correct_l3, correct_l4 = 0, 0, 0, 0
    correct_exact = 0
    total = 0

    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids_R = batch['input_ids_R'].to(device)
        attention_mask_R = batch['attention_mask_R'].to(device)
        graph_x_R = batch['graph_x_R'].to(device)
        graph_adj_R = batch['graph_adj_R'].to(device)
        fp_indices_R = batch['fp_indices_R'].to(device)
        fp_mask_R = batch['fp_mask_R'].to(device)
        fp_dense_R = batch['fp_dense_R'].to(device)

        input_ids_P = batch['input_ids_P'].to(device)
        attention_mask_P = batch['attention_mask_P'].to(device)
        graph_x_P = batch['graph_x_P'].to(device)
        graph_adj_P = batch['graph_adj_P'].to(device)
        fp_indices_P = batch['fp_indices_P'].to(device)
        fp_mask_P = batch['fp_mask_P'].to(device)
        fp_dense_P = batch['fp_dense_P'].to(device)

        labels = batch['labels'].to(device)

        with torch.no_grad():
            l1, l2, l3, l4 = model(
                input_ids_R, attention_mask_R, graph_x_R, graph_adj_R, fp_indices_R, fp_mask_R, fp_dense_R,
                input_ids_P, attention_mask_P, graph_x_P, graph_adj_P, fp_indices_P, fp_mask_P, fp_dense_P,
                targets=None
            )

        pred1 = l1.argmax(dim=-1)
        pred2 = l2.argmax(dim=-1)
        pred3 = l3.argmax(dim=-1)
        pred4 = l4.argmax(dim=-1)

        correct_l1 += (pred1 == labels[:, 0]).sum().item()
        correct_l2 += (pred2 == labels[:, 1]).sum().item()
        correct_l3 += (pred3 == labels[:, 2]).sum().item()
        correct_l4 += (pred4 == labels[:, 3]).sum().item()

        exact_match = (pred1 == labels[:, 0]) & (pred2 == labels[:, 1]) & \
                      (pred3 == labels[:, 2]) & (pred4 == labels[:, 3])
        correct_exact += exact_match.sum().item()

        total += labels.size(0)

    acc_l1 = correct_l1 / total * 100
    acc_l2 = correct_l2 / total * 100
    acc_l3 = correct_l3 / total * 100
    acc_l4 = correct_l4 / total * 100
    acc_exact = correct_exact / total * 100

    print("\n" + "=" * 40)
    print(" Test Set Evaluation Results (Tri-modal Neuro-Symbolic System)")
    print("=" * 40)
    print(f"Level 1 (Main Class) Accuracy:     {acc_l1:.2f}%")
    print(f"Level 2 (Subclass) Accuracy:       {acc_l2:.2f}%")
    print(f"Level 3 (Sub-subclass) Accuracy:   {acc_l3:.2f}%")
    print(f"Level 4 (Serial Number) Accuracy:  {acc_l4:.2f}%")
    print("-" * 40)
    print(f" Exact Match (All Correct):        {acc_exact:.2f}%")
    print("=" * 40 + "\n")

    return acc_exact

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print(f"Error: Cannot find model weights {MODEL_PATH} or label dictionary {LABEL_MAP_PATH}")
        return

    print("Loading Tokenizer and label dictionary...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_maps = pickle.load(f)

    num_classes_list = [len(m) for m in label_maps]

    print("Loading test set...")
    test_dataset = HierarchicalECDataset(TEST_PATH, tokenizer, max_len=512, label_maps=label_maps, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=graph_collate_fn)

    print("Initializing tri-modal neuro-symbolic model...")
    encoder = BioDualModalEncoder(PRETRAINED_PATH, atom_feat_dim=ATOM_FEAT_DIM)
    model = BioEC_EnzymePredictor(encoder, num_classes_list, hidden_dim=512)

    print("Loading model weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    evaluate(model, test_loader, DEVICE)

if __name__ == "__main__":
    main()