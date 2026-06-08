"""
Ignore cofactors with fewer than 5 atoms,
only extract cofactors with more than 20 atoms.
"""

import os
import pickle
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from collections import Counter
import numpy as np


def compute_similarity(mol1, mol2):
    """Calculate the Tanimoto similarity between two molecules"""
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = FingerprintMols.FingerprintMol(mol1)
    fp2 = FingerprintMols.FingerprintMol(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_atom_count(smiles):
    """Calculate the number of atoms in a SMILES string (ignoring hydrogens)"""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    return mol.GetNumAtoms() if mol else 0


def extract_cofactors(data_dir, output_dir, splits=['train', 'valid', 'test'],
                      main_similarity_threshold=0.3, cofactor_similarity_threshold=0.1,
                      min_similarity_gap=0.05, small_molecule_threshold=5,
                      atom_diff_threshold=10, cofactor_atom_max=20):
    """
    Extract cofactors and generate a vocabulary, considering molecule size and dynamic similarity, and adding new constraints.

    Args:
        main_similarity_threshold (float): Minimum similarity for the main reactant. Anything higher is considered a main reactant.
        cofactor_similarity_threshold (float): Maximum similarity for a cofactor.
        min_similarity_gap (float): Minimum similarity gap between the main reactant and the cofactor.
        small_molecule_threshold (int): Atom count threshold for small molecules (used for similarity judgment).
        atom_diff_threshold (int): Threshold for atom count difference between product and reactants. If the difference is smaller than this, skip extracting cofactors.
        cofactor_atom_max (int): Maximum atom count for cofactors (extractable if greater than this value).
    """
    cofactor_candidates = Counter()
    low_similarity_samples = []
    skipped_reactions = 0  # Track the number of skipped reactions

    for split in splits:
        src_file = os.path.join(data_dir, f"src_{split}.txt")
        tgt_file = os.path.join(data_dir, f"tgt_{split}.txt")

        if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
            print(f"Warning: {split} data file missing, skipping")
            continue

        with open(src_file, 'r') as f:
            src_smiles = [line.strip() for line in f if line.strip()]
        with open(tgt_file, 'r') as f:
            tgt_smiles = [line.strip().split('.') for line in f if line.strip()]

        if len(src_smiles) != len(tgt_smiles):
            print(f"Error: {split} data length mismatch")
            continue

        for i, (src, tgt_list) in enumerate(zip(src_smiles, tgt_smiles)):
            src_mol = Chem.MolFromSmiles(src, sanitize=True)
            if src_mol is None:
                print(f"Warning: Invalid product SMILES (index {i}): {src}")
                continue

            if not tgt_list or all(tgt == '' for tgt in tgt_list):
                print(f"Warning: Sample {i} has no valid reactants, skipping")
                continue

            # Calculate atom counts
            src_atom_count = get_atom_count(src)
            tgt_atom_counts = [get_atom_count(tgt) for tgt in tgt_list if tgt]
            total_tgt_atom_count = sum(tgt_atom_counts)

            # Constraint 1: Skip if the difference in atom count between product and reactants is less than 10
            atom_diff = abs(src_atom_count - total_tgt_atom_count)
            if atom_diff < atom_diff_threshold:
                skipped_reactions += 1
                print(f"Skipping sample {i}: Atom count difference {atom_diff} < {atom_diff_threshold}")
                continue

            # Calculate similarities and atom counts
            similarities = []
            atom_counts = []
            tgt_mols = []
            for tgt in tgt_list:
                mol = Chem.MolFromSmiles(tgt, sanitize=True)
                tgt_mols.append(mol)
                if mol is None:
                    print(f"Warning: Invalid reactant SMILES (index {i}): {tgt}")
                    similarities.append(0.0)
                    atom_counts.append(0)
                else:
                    similarities.append(compute_similarity(src_mol, mol))
                    atom_counts.append(mol.GetNumAtoms())

            # Dynamically identify the main reactant
            sorted_similarities = sorted(similarities, reverse=True)
            main_reactant_indices = []
            max_sim = sorted_similarities[0]

            if max_sim >= main_similarity_threshold:
                main_reactant_indices = [j for j, sim in enumerate(similarities)
                                         if sim >= main_similarity_threshold]
            else:
                max_idx = np.argmax(similarities)
                main_reactant_indices = [max_idx]
                if max_sim < 0.25:
                    low_similarity_samples.append((i, src, tgt_list, max_sim))
                    print(f"Warning: Sample {i} has no high-similarity main reactant, taking the highest value ({max_sim:.2f})")

            # Extract cofactors
            for j, (tgt, sim, atom_count) in enumerate(zip(tgt_list, similarities, atom_counts)):
                if tgt == '':
                    continue
                if j not in main_reactant_indices:
                    # Constraint 2: Cofactor atom count is greater than the max threshold (e.g., 15 or 20)
                    if atom_count > cofactor_atom_max:
                        # Meets original conditions: small molecule, low similarity, or large similarity gap
                        if atom_count < small_molecule_threshold or \
                                sim < cofactor_similarity_threshold or \
                                (sorted_similarities[0] - sim >= min_similarity_gap):
                            cofactor_candidates[tgt] += 1
            print(
                f"Sample {i}: src={src}, tgt={tgt_list}, similarities={[round(s, 3) for s in similarities]}, atom_counts={atom_counts}, main_idx={main_reactant_indices}")

    # Generate cofactor vocabulary (occurrence > 5)
    cofactor_vocab = {}
    for idx, (smiles, count) in enumerate(cofactor_candidates.most_common(), 1):
        if count > 6:
            cofactor_vocab[smiles] = f"CF_{idx}"

    # Save vocabulary
    os.makedirs(output_dir, exist_ok=True)
    pkl_file = os.path.join(output_dir, 'cofactor_vocab.pkl')
    txt_file = os.path.join(output_dir, 'cofactor_vocab.txt')

    with open(pkl_file, 'wb') as f:
        pickle.dump(cofactor_vocab, f)

    with open(txt_file, 'w') as f:
        f.write("ID\tSMILES\tCount\n")
        for smiles, count in cofactor_candidates.most_common():
            if smiles in cofactor_vocab:
                f.write(f"{cofactor_vocab[smiles]}\t{smiles}\t{count}\n")

    # Save low similarity samples
    log_file = os.path.join(output_dir, 'low_similarity_samples.txt')
    with open(log_file, 'w') as f:
        f.write("Index\tProduct\tReactants\tMaxSimilarity\n")
        for idx, src, tgt_list, max_sim in low_similarity_samples:
            f.write(f"{idx}\t{src}\t{'.'.join(tgt_list)}\t{max_sim:.2f}\n")

    print(f"Extracted {len(cofactor_vocab)} cofactors, saved to {pkl_file} and {txt_file}")
    print(f"Skipped {skipped_reactions} reactions (atom count difference < {atom_diff_threshold})")
    print("Top 10 common cofactors:")
    print(f"Logged {len(low_similarity_samples)} low similarity samples to {log_file}")
    for smiles, count in cofactor_candidates.most_common(10):
        if smiles in cofactor_vocab:
            print(f"{cofactor_vocab[smiles]}: {smiles} (occurred {count} times)")


def main():
    data_dir = '/data/stu1/ml_project/bioec_retro1/dataset/data_processed/tokenized'
    output_dir = '/data/stu1/ml_project/bioec_retro1/dataset/data_processed/tokenized'
    extract_cofactors(data_dir, output_dir)


if __name__ == "__main__":
    main()