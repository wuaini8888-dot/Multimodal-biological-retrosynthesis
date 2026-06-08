import os
import pickle
from rdkit import Chem
from rdkit import RDLogger
import re

# Block RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Path configurations
BASE_DIR = '/data/stu1/ml_project/bioec_retro1/natural_data/mentor_testset'
VOCAB_PATH = '/data/stu1/ml_project/bioec_retro1/dataset/data_processed/cofactors/cofactor_vocab.pkl'

# Regex pattern for SMILES tokenization
SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
tokenizer_re = re.compile(SMILES_TOKENIZER_PATTERN)


def tokenize_smiles(smiles):
    if not smiles: return ""
    return " ".join(tokenizer_re.findall(smiles))


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        # Set to True here, working perfectly
        if mol: return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except:
        pass
    return None


def process_product_side(smiles_line):
    """Product purification: extract the largest heavy-atom skeleton (>= 5 atoms)"""
    fragments = smiles_line.replace(" ", "").split('.')
    largest_frag = ""
    max_heavy_atoms = -1
    for frag in fragments:
        if not frag: continue
        try:
            mol = Chem.MolFromSmiles(frag)
            if mol:
                h_atoms = mol.GetNumHeavyAtoms()
                if h_atoms > max_heavy_atoms and h_atoms >= 5:
                    max_heavy_atoms = h_atoms
                    # Ultimate fix: changed False to True here!
                    largest_frag = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except:
            pass
    return largest_frag


def process_reactant_side(smiles_line, blacklist):
    """Substrate denoising: remove cofactors and small impurities (< 4 atoms)"""
    fragments = smiles_line.replace(" ", "").split('.')
    kept = []
    for frag in fragments:
        if not frag: continue
        try:
            mol = Chem.MolFromSmiles(frag)
            if mol is None or mol.GetNumHeavyAtoms() < 4:
                continue
        except:
            continue

        canon = canonicalize(frag)
        if canon and canon not in blacklist:
            kept.append(canon)
    kept.sort()
    return ".".join(kept)


def run_cleaning():
    print("Loading cofactor blacklist...")
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    blacklist = {canonicalize(smi) for smi in vocab.keys() if canonicalize(smi)}

    src_in = os.path.join(BASE_DIR, 'mentor_test_src.txt')
    tgt_in = os.path.join(BASE_DIR, 'mentor_test_tgt.txt')
    src_out = os.path.join(BASE_DIR, 'clean_mentor_src.txt')
    tgt_out = os.path.join(BASE_DIR, 'clean_mentor_tgt.txt')

    print(f"Starting to clean 110 data entries...")
    with open(src_in, 'r') as f: src_lines = f.readlines()
    with open(tgt_in, 'r') as f: tgt_lines = f.readlines()

    with open(src_out, 'w') as f_s, open(tgt_out, 'w') as f_t:
        for s, t in zip(src_lines, tgt_lines):
            c_src = process_product_side(s.strip())
            c_tgt = process_reactant_side(t.strip(), blacklist)

            # Even if empty after cleaning, use a placeholder to maintain line alignment
            f_s.write(tokenize_smiles(c_src if c_src else "C") + "\n")
            f_t.write(tokenize_smiles(c_tgt if c_tgt else "C") + "\n")

    print(f"Cleaning complete! Files saved to: {BASE_DIR}")


if __name__ == "__main__":
    run_cleaning()