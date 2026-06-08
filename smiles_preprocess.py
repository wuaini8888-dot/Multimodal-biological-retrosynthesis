import os
import pickle
import re

from rdkit import Chem
import resource
import multiprocessing as mp
import pandas as pd


# SMILES tokenization regular expression
SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
SMILES_REGEX = re.compile(SMILES_TOKENIZER_PATTERN)

def tokenize_smiles(smiles: str) -> list:
    tokens = [token for token in SMILES_REGEX.findall(smiles)]
    return tokens

def build_vocab(smiles_list):
    """Build vocabulary, supporting list or string input"""
    vocab = set()
    for smiles in smiles_list:
        if isinstance(smiles, list):
            for s in smiles:
                tokens = tokenize_smiles(s)
                vocab.update(tokens)
        else:
            tokens = tokenize_smiles(smiles)
            vocab.update(tokens)
    # Add all special tokens, including <unk>
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(list(vocab))
    return {word: idx for idx, word in enumerate(vocab)}

def preprocess_data(data_dir, output_dir, splits=['train', 'valid', 'test']):
    os.makedirs(output_dir, exist_ok=True)

    # Load cofactor vocabulary (txt file)
    cofactor_vocab_file = '/data/stu1/hjq_pycharm_project/A66/data_preprocess/data/cofactors/cofactor_vocab.txt'
    if not os.path.exists(cofactor_vocab_file):
        raise FileNotFoundError(f"Cofactor vocabulary not found: {cofactor_vocab_file}")
    cofactor_df = pd.read_csv(cofactor_vocab_file, sep='\t')
    cofactor_vocab = set(cofactor_df['SMILES'].tolist())

    # Collect all SMILES and filter reactants in tgt
    all_smiles = set()
    all_tokens = set()
    src_smiles_dict = {}
    tgt_smiles_dict = {}
    for split in splits:
        src_file = os.path.join(data_dir, f"src_{split}.txt")
        tgt_file = os.path.join(data_dir, f"tgt_{split}.txt")

        if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
            raise FileNotFoundError(f"Missing data file for {split}")

        with open(src_file, 'r') as f:
            src_smiles = [line.strip() for line in f if line.strip()]
        with open(tgt_file, 'r') as f:
            tgt_smiles = [line.strip().split('.') for line in f if line.strip()]

        if len(src_smiles) != len(tgt_smiles):
            raise ValueError(f"{split} data length mismatch")

        # Filter reactants in tgt: atom count >= 4 and not in the cofactor vocabulary
        filtered_src_smiles = []
        filtered_tgt_smiles = []
        for src, tgt_list in zip(src_smiles, tgt_smiles):
            filtered_tgt = []
            for tgt in tgt_list:
                mol = Chem.MolFromSmiles(tgt, sanitize=True)
                if mol is None:
                    print(f"Invalid SMILES: {tgt}, skipping")
                    continue
                num_atoms = mol.GetNumAtoms()
                if num_atoms >= 4 and tgt not in cofactor_vocab:
                    filtered_tgt.append(tgt)
                else:
                    print(f"Filtered out reactant: tgt={tgt}, atom_count={num_atoms}, is_cofactor={tgt in cofactor_vocab}")

            # Keep all samples, even if tgt is empty after filtering
            filtered_src_smiles.append(src)
            filtered_tgt_smiles.append(filtered_tgt)

        src_smiles_dict[split] = filtered_src_smiles
        tgt_smiles_dict[split] = filtered_tgt_smiles

        # Update SMILES and token sets
        all_smiles.update(filtered_src_smiles)
        for tgt in filtered_tgt_smiles:
            all_smiles.update(tgt)
        all_tokens.update(filtered_src_smiles)
        for tgt in filtered_tgt_smiles:
            all_tokens.update(tgt)

    # Build vocabulary
    all_tokens.add('.')
    vocab = build_vocab(all_tokens)
    with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocab size: {len(vocab)}, saved to {output_dir}/vocab.pkl")

    # Process each split
    for split in splits:
        src_smiles = src_smiles_dict[split]
        tgt_smiles = tgt_smiles_dict[split]

        dataset = []
        for i, (src, tgt_list) in enumerate(zip(src_smiles, tgt_smiles)):
            src_tokens = tokenize_smiles(src)
            tgt_tokens = []
            for j, tgt in enumerate(tgt_list):
                if j > 0:
                    tgt_tokens.append('.')
                tgt_tokens.extend(tokenize_smiles(tgt))

            dataset.append({
                'src_smiles': src,
                'tgt_smiles': tgt_list,
                'src_tokens': src_tokens,
                'tgt_tokens': tgt_tokens,
            })

        # Save dataset as .pkl file
        output_file = os.path.join(output_dir, f"{split}_dataset.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved {split} dataset with {len(dataset)} samples to {output_file}")

        # Save tokenized tokens as .txt files
        src_tokens_txt_file = os.path.join(output_dir, f"tokenized_src_{split}.txt")
        tgt_tokens_txt_file = os.path.join(output_dir, f"tokenized_tgt_{split}.txt")
        with open(src_tokens_txt_file, 'w') as src_f, open(tgt_tokens_txt_file, 'w') as tgt_f:
            for data in dataset:
                src_tokens_str = ' '.join(data['src_tokens'])
                src_f.write(f"{src_tokens_str}\n")
                tgt_tokens_str = ' '.join(data['tgt_tokens']) if data['tgt_tokens'] else ''
                tgt_f.write(f"{tgt_tokens_str}\n")
        print(f"Saved {split} tokenized SMILES to {src_tokens_txt_file} and {tgt_tokens_txt_file}")


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
    mp.set_start_method('spawn')
    data_dir = '/data/stu1/hjq_pycharm_project/A66/data_preprocess/data/ecreact'
    output_dir = '/data/stu1/hjq_pycharm_project/A66/1'
    preprocess_data(data_dir, output_dir)