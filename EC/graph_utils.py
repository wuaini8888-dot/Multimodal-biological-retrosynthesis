import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger

# Block RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Total feature dimension based on the one-hot encoding below
# (44 symbols + 11 degrees + 5 Hs + 6 valences + 1 aromatic = 67)
ATOM_FEAT_DIM = 67


def one_of_k_encoding(x, allowable_set):
    """Maps an input to a one-hot encoding based on an allowable set."""
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps an input to a one-hot encoding, mapping unseen inputs to the last element (Unknown)."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features_58(atom):
    """
    Extracts atomic features using RDKit and converts them to a one-hot feature vector.
    Returns a numpy array of dimension ATOM_FEAT_DIM (67).
    """
    symbol_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                   'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                   'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol_list) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) + \
              [atom.GetIsAromatic()]

    return np.array(results).astype(np.float32)


class RGCNLayer(nn.Module):
    """
    Relational Graph Convolutional Network (RGCN) Layer.
    Processes multi-relational graphs (e.g., single, double, triple, aromatic bonds).
    """

    def __init__(self, in_feat, out_feat, num_relations=4):
        super(RGCNLayer, self).__init__()
        self.num_relations = num_relations
        self.linears = nn.ModuleList([nn.Linear(in_feat, out_feat) for _ in range(num_relations)])

    def forward(self, x, adj_r):
        """
        Forward pass for the RGCN layer.

        Args:
            x (Tensor): Node feature matrix.
            adj_r (Tensor): Multi-relational adjacency matrix.
        """
        out = 0
        for r in range(self.num_relations):
            a_r = adj_r[:, r, :, :]
            x_r = self.linears[r](x)
            out += torch.matmul(a_r, x_r)
        return out


class GNN(nn.Module):
    """
    Graph Neural Network composed of stacked RGCN layers.
    Used for extracting structural embeddings from molecular graphs.
    """

    def __init__(self, in_feat_dim, hidden_dim, num_layers=3, num_relations=4):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(RGCNLayer(in_feat_dim, hidden_dim, num_relations))
        for _ in range(num_layers - 1):
            self.layers.append(RGCNLayer(hidden_dim, hidden_dim, num_relations))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        """
        Forward pass through the stacked RGCN layers with GELU activations.
        """
        for layer in self.layers:
            x = layer(x, adj)
            x = F.gelu(x)
        return self.norm(x)