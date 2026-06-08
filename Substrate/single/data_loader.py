import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from graph_utils import smiles_to_graph, ATOM_FEAT_DIM
# Import RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class Vocab:
    def __init__(self, vocab_file):
        self.token2id = {'<unk>': 0, '<blank>': 1, '<s>': 2, '</s>': 3}
        self.id2token = {0: '<unk>', 1: '<blank>', 2: '<s>', 3: '</s>'}
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        token = parts[0]
                        if token not in self.token2id:
                            idx = len(self.token2id)
                            self.token2id[token] = idx
                            self.id2token[idx] = token
        except FileNotFoundError:
            print(f"Warning: Vocab file {vocab_file} not found. Model will use only special tokens.")

    def __len__(self):
        return len(self.token2id)

    def encode(self, tokens, add_bos=False, add_eos=False):
        ids = [self.token2id.get(t, UNK_IDX) for t in tokens]
        if add_bos: ids = [BOS_IDX] + ids
        if add_eos: ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            idx = i.item() if hasattr(i, 'item') else i
            token = self.id2token.get(idx, '<unk>')
            if token not in ['<blank>', '<s>', '</s>']: tokens.append(token)
        return tokens


def get_morgan_fingerprint(smiles, fp_dim=2048):
    """Extract the Morgan fingerprint of the target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros((fp_dim,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    return np.array(fp, dtype=np.float32)


class ReactionDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab, max_nodes=150):
        self.max_nodes = max_nodes
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        with open(src_path, 'r', encoding='utf-8') as f:
            self.src_lines = [line.strip() for line in f.readlines()]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            self.tgt_lines = [line.strip() for line in f.readlines()]

    def __len__(self): return len(self.src_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx]
        tgt_line = self.tgt_lines[idx]

        src_tokens = src_line.split()
        tgt_tokens = tgt_line.split()

        src_ids = self.src_vocab.encode(src_tokens, add_bos=False, add_eos=False)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens, add_bos=True, add_eos=True)

        raw_smiles = "".join(src_tokens)
        graph_x, graph_adj = smiles_to_graph(raw_smiles, self.max_nodes)

        # Extract fingerprint
        fp_array = get_morgan_fingerprint(raw_smiles)

        return src_ids, tgt_ids, graph_x, graph_adj, fp_array


def collate_fn(batch):
    # Add fp_batch
    src_batch, tgt_batch, graph_x_batch, graph_adj_batch, fp_batch = zip(*batch)

    src_tensors = [torch.tensor(s, dtype=torch.long) for s in src_batch]
    tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in tgt_batch]

    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=PAD_IDX)

    graph_x_padded = torch.stack([torch.tensor(gx, dtype=torch.float) for gx in graph_x_batch])
    graph_adj_padded = torch.stack([torch.tensor(ga, dtype=torch.float) for ga in graph_adj_batch])

    # Stack fingerprints
    fp_padded = torch.stack([torch.tensor(f, dtype=torch.float) for f in fp_batch])

    return src_padded, tgt_padded, graph_x_padded, graph_adj_padded, fp_padded


def build_weighted_loader(vocab_src, vocab_tgt, config):
    max_n = config.get('max_graph_nodes', 150)

    print("Loading USPTO...")
    ds_1 = ReactionDataset(config['uspto_src'], config['uspto_tgt'], vocab_src, vocab_tgt, max_nodes=max_n)
    print("Loading Bio...")
    ds_2 = ReactionDataset(config['bio_src'], config['bio_tgt'], vocab_src, vocab_tgt, max_nodes=max_n)

    full_dataset = torch.utils.data.ConcatDataset([ds_1, ds_2])

    uspto_weight = float(config.get('uspto_weight', 1.0))
    bio_weight = float(config.get('bio_weight', 1.0))

    weights = [uspto_weight] * len(ds_1) + [bio_weight] * len(ds_2)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return DataLoader(full_dataset, batch_size=config['batch_size'],
                      sampler=sampler, collate_fn=collate_fn,
                      num_workers=4, pin_memory=True)