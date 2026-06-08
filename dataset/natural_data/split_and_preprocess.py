import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import re

# Block RDKit warnings
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Core paths and configurations
# ==========================================
BASE_DIR = '/data/stu1/ml_project/bioec_retro1/natural_data'
# Updated to your recently uploaded cleanly washed file
EXCEL_PATH = os.path.join(BASE_DIR, 'SMILES_EC1.xlsx')
VOCAB_PATH = '/data/stu1/ml_project/bioec_retro1/dataset/data_processed/cofactors/cofactor_vocab.pkl'

CLEAN_DIR = os.path.join(BASE_DIR, 'mentor_testset', 'category_processed', 'clean')
RAW_DIR = os.path.join(BASE_DIR, 'mentor_testset', 'category_processed', 'raw')

CATEGORY_MAP = {'酚酸类': 'phenols', '萜类': 'terpenes', '黄酮类': 'flavonoids', '甾体类': 'steroids'}

# Senior's ancestral tokenization regex
SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
tokenizer_re = re.compile(SMILES_TOKENIZER_PATTERN)


def tokenize_smiles(smiles):
    if not smiles or pd.isna(smiles) or smiles == "C": return "C"
    return " ".join(tokenizer_re.findall(str(smiles).replace(" ", "")))


def parse_excel_cell(cell_str):
    """[Ultimate fix]: Intelligently strip text, and use chemical visual regex to safely split true and false plus signs"""
    if pd.isna(cell_str): return None
    s = str(cell_str)

    # 1. Clean up various bizarre line breaks and residual prefixes (e.g., "\n" or "\r")
    # Forcefully convert newline characters to plus sign concatenation
    s = s.replace("\n", " + ").replace("\r", " + ")

    # 2. Core regex weapon: split '+', but ignore '+' inside brackets (e.g., perfectly avoids [NH3+])
    # Logic: Find '+', provided there is no unclosed ']' from its right side to the end
    raw_frags = re.split(r'\+(?![^\[]*\])', s)

    valid_smiles = []
    for f in raw_frags:
        # Remove any potentially remaining text prefixes
        if ":" in f: f = f.split(":")[-1]
        f = f.strip().replace(" ", "")
        if not f: continue

        mol = Chem.MolFromSmiles(f)
        if mol:
            valid_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))

    if not valid_smiles: return None
    return ".".join(sorted(valid_smiles))


def process_product_clean(canon_smiles):
    """Extract the largest skeleton"""
    if not canon_smiles: return "C"
    frags = canon_smiles.split('.')
    max_h_atoms = -1
    largest = ""
    for f in frags:
        mol = Chem.MolFromSmiles(f)
        if mol:
            ha = mol.GetNumHeavyAtoms()
            # Extract as main product skeleton if heavy atoms >= 5
            if ha > max_h_atoms and ha >= 5:
                max_h_atoms = ha
                largest = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return largest if largest else "C"


def process_reactant_clean(canon_smiles, blacklist):
    """Remove coenzymes and small impurities found in the blacklist"""
    if not canon_smiles: return "C"
    frags = canon_smiles.split('.')
    kept = []
    for f in frags:
        mol = Chem.MolFromSmiles(f)
        # Only keep if heavy atoms >= 4 and not in the coenzyme blacklist
        if mol and mol.GetNumHeavyAtoms() >= 4:
            if f not in blacklist:
                kept.append(f)
    return ".".join(sorted(kept)) if kept else "C"


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

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

    print(f"Reading Excel: {EXCEL_PATH}")
    try:
        excel_data = pd.read_excel(EXCEL_PATH, sheet_name=None)
    except Exception as e:
        print(f" Failed to read Excel, please check if the file exists: {e}")
        return

    for sheet_name, prefix in CATEGORY_MAP.items():
        if sheet_name not in excel_data: continue
        df = excel_data[sheet_name]

        prod_col = next((col for col in df.columns if '产物SMILES' in col), None)
        reac_col = next((col for col in df.columns if '底物SMILES' in col), None)
        if not prod_col or not reac_col: continue

        clean_src_list, clean_tgt_list, raw_src_list, raw_tgt_list = [], [], [], []

        for idx, row in df.iterrows():
            # 1. Parse and read (fully retain true reactants and coenzymes, unify 3D order) -> Fair Raw
            canon_raw_prod = parse_excel_cell(row[prod_col])
            canon_raw_reac = parse_excel_cell(row[reac_col])
            raw_src_list.append(tokenize_smiles(canon_raw_prod if canon_raw_prod else "C"))
            raw_tgt_list.append(tokenize_smiles(canon_raw_reac if canon_raw_reac else "C"))

            # 2. Combat noise (precisely cut out blacklisted coenzymes and small salt molecules) -> Clean
            clean_src = process_product_clean(canon_raw_prod)
            clean_tgt = process_reactant_clean(canon_raw_reac, blacklist)
            clean_src_list.append(tokenize_smiles(clean_src if clean_src else "C"))
            clean_tgt_list.append(tokenize_smiles(clean_tgt if clean_tgt else "C"))

        # Write to files
        with open(os.path.join(CLEAN_DIR, f"{prefix}_src.txt"), 'w') as f:
            f.write("\n".join(clean_src_list) + "\n")
        with open(os.path.join(CLEAN_DIR, f"{prefix}_tgt.txt"), 'w') as f:
            f.write("\n".join(clean_tgt_list) + "\n")
        with open(os.path.join(RAW_DIR, f"raw_{prefix}_src.txt"), 'w') as f:
            f.write("\n".join(raw_src_list) + "\n")
        with open(os.path.join(RAW_DIR, f"raw_{prefix}_tgt.txt"), 'w') as f:
            f.write("\n".join(raw_tgt_list) + "\n")

        print(f" {sheet_name} generated!")

    print("\n All parsing complete, you can finally calculate scores with peace of mind!")


if __name__ == '__main__':
    main()