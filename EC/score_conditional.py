import os
import argparse
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer

# Block RDKit warnings
RDLogger.DisableLog('rdApp.*')

from ec_model import BioEC_EnzymePredictor, BioDualModalEncoder
from graph_utils import ATOM_FEAT_DIM
from ec_data import HierarchicalECDataset


def canonicalize_smiles(smi):
    if not smi or not isinstance(smi, str): return None
    try:
        smi = smi.replace(" ", "")
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except:
        return None


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f" Initializing ultimate conditional probability validation (Device: {device})")

    df_clean = pd.read_csv(args.clean_csv)
    with open(args.predictions, 'r') as f:
        pred_lines = f.readlines()

    total = len(df_clean)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    with open(args.label_maps, 'rb') as f:
        label_maps = pickle.load(f)

    num_classes_list = [len(m) for m in label_maps]
    inv_label_maps = [{v: k for k, v in m.items()} for m in label_maps]

    encoder = BioDualModalEncoder(args.pretrained_path, atom_feat_dim=ATOM_FEAT_DIM)
    ec_model = BioEC_EnzymePredictor(encoder, num_classes_list, hidden_dim=512)
    ec_model.load_state_dict(torch.load(args.ec_ckpt, map_location=device))
    ec_model.to(device)
    ec_model.eval()

    dummy_dataset = HierarchicalECDataset(args.clean_csv, tokenizer, max_len=512, label_maps=label_maps, is_train=False)

    # Statistics counters
    substrate_correct_count = 0
    ec_exact_match_in_subset = 0
    ec_l1_in_subset = 0
    ec_l2_in_subset = 0
    ec_l3_in_subset = 0
    ec_l4_in_subset = 0

    print("\n" + "=" * 50)
    print(" Starting conditional test: P(EC correct | Substrate Top-1 correct)")
    print("=" * 50)

    for i in tqdm(range(total)):
        row_clean = df_clean.iloc[i]

        # Achiral canonical for substrate matching
        target_reac_clean = canonicalize_smiles(str(row_clean['Reactiants']))

        # RAW data retaining 3D chiral features to be fed into the EC model
        target_prod_raw = str(row_clean['Products']).replace(" ", "")
        target_reac_raw = str(row_clean['Reactiants']).replace(" ", "")

        true_ec = str(row_clean['EC'])
        true_ec_parts = true_ec.split('.')

        # ==========================================
        # Step 1: Check if substrate Top-1 prediction is correct
        # ==========================================
        parts = pred_lines[i].strip().split('\t')
        pred_top1 = parts[1].replace(" ", "") if len(parts) > 1 else ""
        pred_top1_canon = canonicalize_smiles(pred_top1)

        # We only calculate EC accuracy if the substrate is correct
        if pred_top1_canon == target_reac_clean:
            substrate_correct_count += 1

            # ==========================================
            # Step 2: Substrate is correct, feed into EC model for validation
            # ==========================================
            inputs_R = tokenizer(target_reac_raw, truncation=True, padding='max_length', max_length=512,
                                 return_tensors='pt').to(device)
            inputs_P = tokenizer(target_prod_raw, truncation=True, padding='max_length', max_length=512,
                                 return_tensors='pt').to(device)

            x_R, adj_R = dummy_dataset.smiles_to_graph(target_reac_raw)
            x_P, adj_P = dummy_dataset.smiles_to_graph(target_prod_raw)
            fp_dense_R, fp_indices_R = dummy_dataset.smiles_to_fp(target_reac_raw)
            fp_dense_P, fp_indices_P = dummy_dataset.smiles_to_fp(target_prod_raw)

            fp_mask_R = torch.zeros(fp_indices_R.size(0), dtype=torch.bool).unsqueeze(0).to(device)
            fp_mask_P = torch.zeros(fp_indices_P.size(0), dtype=torch.bool).unsqueeze(0).to(device)

            with torch.no_grad():
                l1, l2, l3, l4 = ec_model(
                    inputs_R['input_ids'], inputs_R['attention_mask'], x_R.unsqueeze(0).to(device),
                    adj_R.unsqueeze(0).to(device),
                    fp_indices_R.unsqueeze(0).to(device), fp_mask_R, fp_dense_R.unsqueeze(0).to(device),
                    inputs_P['input_ids'], inputs_P['attention_mask'], x_P.unsqueeze(0).to(device),
                    adj_P.unsqueeze(0).to(device),
                    fp_indices_P.unsqueeze(0).to(device), fp_mask_P, fp_dense_P.unsqueeze(0).to(device),
                    targets=None
                )

            p_l1 = inv_label_maps[0].get(l1.argmax(dim=-1).item(), '0')
            p_l2 = inv_label_maps[1].get(l2.argmax(dim=-1).item(), '0')
            p_l3 = inv_label_maps[2].get(l3.argmax(dim=-1).item(), '0')
            p_l4 = inv_label_maps[3].get(l4.argmax(dim=-1).item(), '0')

            pred_ec = f"{p_l1}.{p_l2}.{p_l3}.{p_l4}"

            if p_l1 == true_ec_parts[0]: ec_l1_in_subset += 1
            if p_l2 == true_ec_parts[1]: ec_l2_in_subset += 1
            if p_l3 == true_ec_parts[2]: ec_l3_in_subset += 1
            if p_l4 == true_ec_parts[3]: ec_l4_in_subset += 1
            if pred_ec == true_ec: ec_exact_match_in_subset += 1

    print("\n" + "=" * 50)
    print("  Ultimate Conditional Probability Validation Results: P(EC | Substrate)")
    print("=" * 50)
    print(f"Total samples in golden dataset: {total}")
    print(f"-> Substrate Top-1 correct predictions (Denominator): {substrate_correct_count} ({substrate_correct_count / total:.2%})")

    if substrate_correct_count > 0:
        print("-" * 50)
        print("Among the reactions with correctly predicted substrates, the conditional accuracy of the EC model is:")
        print(f"Level 1 Accuracy:   {ec_l1_in_subset / substrate_correct_count:.2%}")
        print(f"Level 2 Accuracy:   {ec_l2_in_subset / substrate_correct_count:.2%}")
        print(f"Level 3 Accuracy:   {ec_l3_in_subset / substrate_correct_count:.2%}")
        print(f"Level 4 Accuracy:   {ec_l4_in_subset / substrate_correct_count:.2%}")
        print(f"▶ EC Exact Match:   {ec_exact_match_in_subset / substrate_correct_count:.2%} ◀")
        print("-" * 50)
        print(
            f"Mathematical verification: {substrate_correct_count / total:.2%} (Substrate) x {ec_exact_match_in_subset / substrate_correct_count:.2%} (Conditional EC) = {(substrate_correct_count / total) * (ec_exact_match_in_subset / substrate_correct_count):.2%} (Total Joint Accuracy)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    base_dir = '/data/stu1/ml_project