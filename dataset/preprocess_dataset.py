import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Block RDKit warnings
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Core path configurations
# ==========================================
BASE_DIR = '/data/stu1/ml_project/bioec_retro1/dataset/EC'
VOCAB_PATH = '/data/stu1/ml_project/bioec_retro1/dataset/data_processed/cofactors/cofactor_vocab.pkl'
# To avoid overwriting your original data, generated new files are placed in a new folder
OUTPUT_DIR = os.path.join(BASE_DIR, 'clean_processed_csv')


def process_product_clean(canon_smiles):
    """Extract product core skeleton: keep only the main structure with heavy atoms >= 5, no tokenization"""
    if pd.isna(canon_smiles) or not canon_smiles: return "C"
    frags = str(canon_smiles).split('.')
    max_h_atoms = -1
    largest = ""
    for f in frags:
        f = f.strip()
        if not f: continue
        mol = Chem.MolFromSmiles(f)
        if mol:
            ha = mol.GetNumHeavyAtoms()
            if ha > max_h_atoms and ha >= 5:
                max_h_atoms = ha
                # Strictly retain stereochemistry, no spaces added
                largest = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return largest if largest else "C"


def process_reactant_clean(canon_smiles, blacklist):
    """Substrate denoising: precisely cut out cofactors like ATP, NADPH and small impurities, no tokenization"""
    if pd.isna(canon_smiles) or not canon_smiles: return "C"
    frags = str(canon_smiles).split('.')
    kept = []
    for f in frags:
        f = f.strip()
        if not f: continue
        mol = Chem.MolFromSmiles(f)
        # Discard small impurities like water and protons with heavy atoms < 4
        if mol and mol.GetNumHeavyAtoms() >= 4:
            canon_f = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            # If not in the cofactor blacklist, keep it
            if canon_f not in blacklist:
                kept.append(canon_f)
    # Reassemble multiple substrates in alphabetical order
    return ".".join(sorted(kept)) if kept else "C"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and standardize cofactor blacklist
    print("Loading cofactor blacklist...")
    try:
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        blacklist = set()
        for smi in vocab.keys():
            mol = Chem.MolFromSmiles(smi)
            if mol: blacklist.add(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
    except Exception as e:
        print(f" Failed to load blacklist: {e}")
        return

    # 2. Process the three files sequentially
    splits = ['train', 'valid', 'test']
    for split in splits:
        csv_file = os.path.join(BASE_DIR, f"{split}.csv")
        if not os.path.exists(csv_file):
            print(f" File not found: {csv_file}")
            continue

        print(f"Cleaning and reconstructing: {csv_file} ...")
        df = pd.read_csv(csv_file)

        # Dynamically capture column names to prevent spelling differences (your original table misspelled it as Reactiants)
        reac_col = next((col for col in df.columns if 'React' in col), 'Reactiants')
        prod_col = next((col for col in df.columns if 'Prod' in col), 'Products')
        ec_col = next((col for col in df.columns if 'EC' in col), 'EC')

        new_rows = []
        for idx, row in df.iterrows():
            raw_prod = row.get(prod_col, "")
            raw_reac = row.get(reac_col, "")
            ec_val = row.get(ec_col, "")

            # Purification calculation
            clean_prod = process_product_clean(raw_prod)
            clean_reac = process_reactant_clean(raw_reac, blacklist)

            # Maintain original dictionary mapping relationship
            new_rows.append({
                reac_col: clean_reac,
                ec_col: ec_val,
                prod_col: clean_prod
            })

        # Generate a new DataFrame and constrain the column order to be exactly the same as the original table
        out_df = pd.DataFrame(new_rows)
        out_df = out_df[[reac_col, ec_col, prod_col]]

        # 3. Output as a CSV file with the same name
        out_path = os.path.join(OUTPUT_DIR, f"{split}.csv")
        out_df.to_csv(out_path, index=False)
        print(f" Generation complete: {out_path} (Retained original CSV format, no tokenization, total {len(out_df)} rows)")


if __name__ == '__main__':
    main()