import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from graph_utils import GNN
import math


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, seq_emb, pad_mask):
        attn_scores = self.pooling(seq_emb).squeeze(-1)
        attn_scores = attn_scores.masked_fill(pad_mask, -1e4)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_pooled = torch.sum(seq_emb * attn_weights.unsqueeze(-1), dim=1)
        return attn_pooled


class BioDualModalEncoder(nn.Module):
    def __init__(self, pretrained_path, d_model=512, nhead=8, num_layers=4, atom_feat_dim=None):
        super().__init__()
        self.d_model = d_model
        print(f"Loading ChemBERTa from {pretrained_path} ...")
        config = AutoConfig.from_pretrained(pretrained_path)
        self.text_backbone = AutoModel.from_pretrained(pretrained_path)
        self.text_proj = nn.Linear(config.hidden_size, d_model)
        self.gnn = GNN(atom_feat_dim, d_model, num_layers=3)

        # Modality expansion: vocabulary is 2048 (0~2047), +1 for Padding (2048)
        self.fp_emb = nn.Embedding(2049, d_model, padding_idx=2048)

        # Type embeddings for the three modalities (0=Text, 1=Graph, 2=Fingerprint)
        self.type_emb = nn.Embedding(3, d_model)
        self.pre_fusion_norm = nn.LayerNorm(d_model)

        self.modality_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=2
        )
        self.attention_pooling = AttentionPooling(d_model)

    def forward(self, input_ids, attention_mask, graph_x, graph_adj, fp_indices, fp_mask):
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.text_proj(text_outputs.last_hidden_state)
        text_emb = text_emb * math.sqrt(self.d_model)
        text_emb = text_emb + self.type_emb(torch.tensor(0, device=text_emb.device))

        graph_emb = self.gnn(graph_x, graph_adj)
        graph_emb = graph_emb * math.sqrt(self.d_model)
        graph_emb = graph_emb + self.type_emb(torch.tensor(1, device=graph_emb.device))

        # Fingerprint substructure embedding
        fp_token_emb = self.fp_emb(fp_indices)
        fp_token_emb = fp_token_emb * math.sqrt(self.d_model)
        fp_token_emb = fp_token_emb + self.type_emb(torch.tensor(2, device=fp_token_emb.device))

        text_pad_mask = (attention_mask == 0)
        graph_pad_mask = (graph_x.abs().sum(dim=-1) == 0)

        # Three-modality sequence concatenation
        fusion_pad_mask = torch.cat([text_pad_mask, graph_pad_mask, fp_mask], dim=1)
        fused_emb = torch.cat([text_emb, graph_emb, fp_token_emb], dim=1)

        fused_emb = self.pre_fusion_norm(fused_emb)
        fused_emb = self.modality_fusion(fused_emb, src_key_padding_mask=fusion_pad_mask)
        attn_pooled = self.attention_pooling(fused_emb, fusion_pad_mask)

        return attn_pooled, fused_emb, fusion_pad_mask


class BioEC_EnzymePredictor(nn.Module):
    def __init__(self, encoder, num_classes_list, hidden_dim=512, nhead=8):
        super().__init__()
        self.encoder = encoder

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.norm_R = nn.LayerNorm(hidden_dim)
        self.norm_P = nn.LayerNorm(hidden_dim)
        self.delta_pooling = AttentionPooling(hidden_dim)

        # Expert system absolute difference encoder (2048-dim -> 512-dim)
        self.fp_delta_proj = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Ultimate feature fusion head: receives 5 sets of features [feat_R, feat_P, delta_R, delta_P, fp_delta] (512 * 5 = 2560)
        self.reaction_proj = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.head1 = nn.Linear(hidden_dim, num_classes_list[0])
        self.emb1 = nn.Embedding(num_classes_list[0], hidden_dim)
        self.head2 = nn.Linear(hidden_dim * 2, num_classes_list[1])
        self.emb2 = nn.Embedding(num_classes_list[1], hidden_dim)
        self.head3 = nn.Linear(hidden_dim * 2, num_classes_list[2])
        self.emb3 = nn.Embedding(num_classes_list[2], hidden_dim)
        self.head4 = nn.Linear(hidden_dim * 2, num_classes_list[3])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids_R, attention_mask_R, graph_x_R, graph_adj_R, fp_indices_R, fp_mask_R, fp_dense_R,
                input_ids_P, attention_mask_P, graph_x_P, graph_adj_P, fp_indices_P, fp_mask_P, fp_dense_P,
                targets=None):
        # 1. Three-modality feature extraction
        feat_R, seq_R, mask_R = self.encoder(input_ids_R, attention_mask_R, graph_x_R, graph_adj_R, fp_indices_R,
                                             fp_mask_R)
        feat_P, seq_P, mask_P = self.encoder(input_ids_P, attention_mask_P, graph_x_P, graph_adj_P, fp_indices_P,
                                             fp_mask_P)

        # 2. Neural network microscopic topology alignment (Flexible difference)
        aligned_R, _ = self.cross_attn(query=seq_R, key=seq_P, value=seq_P, key_padding_mask=mask_P)
        aligned_R = self.norm_R(seq_R + aligned_R)
        aligned_P, _ = self.cross_attn(query=seq_P, key=seq_R, value=seq_R, key_padding_mask=mask_R)
        aligned_P = self.norm_P(seq_P + aligned_P)

        delta_seq_R = aligned_R - seq_R
        delta_seq_P = aligned_P - seq_P
        delta_R = self.delta_pooling(delta_seq_R, mask_R)
        delta_P = self.delta_pooling(delta_seq_P, mask_P)

        # 3. Expert system fingerprint absolute difference (Rigid difference)
        delta_fp = fp_dense_P - fp_dense_R
        fp_delta_feat = self.fp_delta_proj(delta_fp)

        # 4. Ultimate grand unified concatenation and dimensionality reduction
        concat_feat = torch.cat([feat_R, feat_P, delta_R, delta_P, fp_delta_feat], dim=-1)
        features = self.reaction_proj(concat_feat)

        # 5. Cascaded inference
        logits1 = self.head1(features)
        pred1 = targets[:, 0] if self.training and targets is not None else logits1.argmax(dim=-1).detach()

        feat_l2 = torch.cat([features, self.emb1(pred1)], dim=-1)
        feat_l2 = self.dropout(self.relu(feat_l2))
        logits2 = self.head2(feat_l2)
        pred2 = targets[:, 1] if self.training and targets is not None else logits2.argmax(dim=-1).detach()

        feat_l3 = torch.cat([features, self.emb2(pred2)], dim=-1)
        feat_l3 = self.dropout(self.relu(feat_l3))
        logits3 = self.head3(feat_l3)
        pred3 = targets[:, 2] if self.training and targets is not None else logits3.argmax(dim=-1).detach()

        feat_l4 = torch.cat([features, self.emb3(pred3)], dim=-1)
        feat_l4 = self.dropout(self.relu(feat_l4))
        logits4 = self.head4(feat_l4)

        return logits1, logits2, logits3, logits4