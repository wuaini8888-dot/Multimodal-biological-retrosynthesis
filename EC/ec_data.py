import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from graph_utils import ATOM_FEAT_DIM, get_atom_features_58


class HierarchicalECDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=512, label_maps=None, is_train=True):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.ec_parts = self._process_ec_labels()

        if is_train and label_maps is None:
            self.label_maps = self._build_label_maps()
        else:
            self.label_maps = label_maps

    def __len__(self):
        return len(self.data)

    def _process_ec_labels(self):
        ec_list = self.data['EC'].astype(str).tolist()
        parsed_ecs = []
        for ec in ec_list:
            parts = ec.split('.')
            if len(parts) != 4:
                parts = parts + ['0'] * (4 - len(parts))
            parsed_ecs.append(parts)
        return parsed_ecs

    def _build_label_maps(self):
        maps = [{}, {}, {}, {}]
        levels = list(zip(*self.ec_parts))
        for i in range(4):
            unique_labels = sorted(list(set(levels[i])))
            for idx, lbl in enumerate(unique_labels):
                maps[i][lbl] = idx
        return maps

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        num_relations = 4
        if mol is None:
            x = np.zeros((1, ATOM_FEAT_DIM), dtype=np.float32)
            adj = np.zeros((num_relations, 1, 1), dtype=np.float32)
            for r in range(num_relations):
                adj[r, 0, 0] = 1.0
            return torch.tensor(x), torch.tensor(adj)

        atoms = mol.GetAtoms()
        num_atoms = len(atoms)
        x = np.zeros((num_atoms, ATOM_FEAT_DIM), dtype=np.float32)
        for i, atom in enumerate(atoms):
            x[i] = get_atom_features_58(atom)

        adj = np.zeros((num_relations, num_atoms, num_atoms), dtype=np.float32)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            if bond_type == Chem.rdchem.BondType.SINGLE:
                rel_idx = 0
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                rel_idx = 1
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                rel_idx = 2
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                rel_idx = 3
            else:
                rel_idx = 0

            adj[rel_idx, i, j] = 1.0
            adj[rel_idx, j, i] = 1.0

        for r in range(num_relations):
            adj[r] = adj[r] + np.eye(num_atoms)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32)

    def smiles_to_fp(self, smiles):
        """ Extract the dense representation and activated indices of the 2048-dimensional Morgan fingerprint """
        mol = Chem.MolFromSmiles(smiles)
        num_bits = 2048
        if mol is None:
            return torch.zeros(num_bits, dtype=torch.float32), torch.tensor([], dtype=torch.long)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=num_bits)
        fp_dense = torch.tensor(list(fp), dtype=torch.float32)
        fp_indices = torch.tensor(list(fp.GetOnBits()), dtype=torch.long)  # Extract the position indices where the bit is 1
        return fp_dense, fp_indices

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        reactant_smiles = str(row['Reactiants'])
        product_smiles = str(row['Products'])

        # 1. Text modality
        inputs_R = self.tokenizer(reactant_smiles, truncation=True, padding='max_length', max_length=self.max_len,
                                  return_tensors='pt')
        inputs_P = self.tokenizer(product_smiles, truncation=True, padding='max_length', max_length=self.max_len,
                                  return_tensors='pt')

        # 2. Graph modality
        x_R, adj_R = self.smiles_to_graph(reactant_smiles)
        x_P, adj_P = self.smiles_to_graph(product_smiles)

        # 3. Fingerprint modality (New)
        fp_dense_R, fp_indices_R = self.smiles_to_fp(reactant_smiles)
        fp_dense_P, fp_indices_P = self.smiles_to_fp(product_smiles)

        item = {
            'input_ids_R': inputs_R['input_ids'].squeeze(0),
            'attention_mask_R': inputs_R['attention_mask'].squeeze(0),
            'graph_x_R': x_R,
            'graph_adj_R': adj_R,
            'fp_dense_R': fp_dense_R,
            'fp_indices_R': fp_indices_R,

            'input_ids_P': inputs_P['input_ids'].squeeze(0),
            'attention_mask_P': inputs_P['attention_mask'].squeeze(0),
            'graph_x_P': x_P,
            'graph_adj_P': adj_P,
            'fp_dense_P': fp_dense_P,
            'fp_indices_P': fp_indices_P,
        }

        ec_parts = self.ec_parts[idx]
        labels = [self.label_maps[i].get(ec_parts[i], 0) for i in range(4)]
        item['labels'] = torch.tensor(labels, dtype=torch.long)
        return item


def graph_collate_fn(batch):
    item = {
        'input_ids_R': torch.stack([b['input_ids_R'] for b in batch]),
        'attention_mask_R': torch.stack([b['attention_mask_R'] for b in batch]),
        'input_ids_P': torch.stack([b['input_ids_P'] for b in batch]),
        'attention_mask_P': torch.stack([b['attention_mask_P'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'fp_dense_R': torch.stack([b['fp_dense_R'] for b in batch]),
        'fp_dense_P': torch.stack([b['fp_dense_P'] for b in batch])
    }

    # ------------------ Handle Fingerprint Index Padding ------------------
    def pad_fp_indices(indices_list, pad_idx=2048):
        max_len = max([idx.size(0) for idx in indices_list])
        max_len = max(max_len, 1)  # Prevent empty sequences
        padded_idx, masks = [], []
        for idx in indices_list:
            pad_len = max_len - idx.size(0)
            padded_idx.append(torch.cat([idx, torch.full((pad_len,), pad_idx, dtype=torch.long)]))
            masks.append(torch.cat([torch.zeros(idx.size(0), dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)]))
        return torch.stack(padded_idx), torch.stack(masks)

    item['fp_indices_R'], item['fp_mask_R'] = pad_fp_indices([b['fp_indices_R'] for b in batch])
    item['fp_indices_P'], item['fp_mask_P'] = pad_fp_indices([b['fp_indices_P'] for b in batch])
    # ----------------------------------------------------------------------

    # ------------------ Handle Multi-relational Graph Padding ------------------
    max_nodes_R = max([b['graph_x_R'].shape[0] for b in batch])
    max_nodes_P = max([b['graph_x_P'].shape[0] for b in batch])
    num_relations = 4

    batch_x_R, batch_adj_R, batch_x_P, batch_adj_P = [], [], [], []

    for b in batch:
        n_R = b['graph_x_R'].shape[0]
        x_R = torch.cat([b['graph_x_R'], torch.zeros(max_nodes_R - n_R, ATOM_FEAT_DIM)], dim=0)
        adj_R = torch.zeros(num_relations, max_nodes_R, max_nodes_R)
        adj_R[:, :n_R, :n_R] = b['graph_adj_R']
        batch_x_R.append(x_R)
        batch_adj_R.append(adj_R)

        n_P = b['graph_x_P'].shape[0]
        x_P = torch.cat([b['graph_x_P'], torch.zeros(max_nodes_P - n_P, ATOM_FEAT_DIM)], dim=0)
        adj_P = torch.zeros(num_relations, max_nodes_P, max_nodes_P)
        adj_P[:, :n_P, :n_P] = b['graph_adj_P']
        batch_x_P.append(x_P)
        batch_adj_P.append(adj_P)

    item['graph_x_R'] = torch.stack(batch_x_R)
    item['graph_adj_R'] = torch.stack(batch_adj_R)
    item['graph_x_P'] = torch.stack(batch_x_P)
    item['graph_adj_P'] = torch.stack(batch_adj_P)

    return item