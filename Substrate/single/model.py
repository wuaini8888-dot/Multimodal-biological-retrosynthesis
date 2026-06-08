import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# ================= 1. GCN Base Components =================
class RelationalGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations=4, dropout=0.1):
        super(RelationalGCNLayer, self).__init__()
        self.num_relations = num_relations
        self.proj_weights = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        nn.init.xavier_uniform_(self.proj_weights)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.size()
        output = torch.zeros(batch_size, num_nodes, self.proj_weights.size(2)).to(x.device)
        for r in range(self.num_relations):
            h_r = torch.matmul(x, self.proj_weights[r])
            m_r = torch.bmm(adj[:, r], h_r)
            output += m_r
        output = self.act(self.norm(output))
        return self.dropout(output)


class GraphEncoder(nn.Module):
    def __init__(self, atom_feat_dim, d_model, num_layers=3, dropout=0.1):
        super(GraphEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(RelationalGCNLayer(atom_feat_dim, d_model, num_relations=4, dropout=dropout))
        for _ in range(num_layers - 1):
            self.layers.append(RelationalGCNLayer(d_model, d_model, num_relations=4, dropout=dropout))

    def forward(self, x, adj):
        for layer in self.layers: x = layer(x, adj)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ================= 2. Deep Fusion Layer Components =================

# Restore original design 1: Fingerprint gated fusion layer (fully preserving your original logic)
class FingerprintGateFusion(nn.Module):
    def __init__(self, fp_dim=2048, d_model=512, dropout=0.1):
        super(FingerprintGateFusion, self).__init__()
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(d_model)

        nn.init.constant_(self.gate_linear.bias, -10.0)
        nn.init.xavier_uniform_(self.gate_linear.weight)

    def forward(self, text_emb, fp):
        fp_mapped = self.fp_proj(fp).unsqueeze(1)
        fp_expanded = fp_mapped.expand(-1, text_emb.size(1), -1)

        concat_feat = torch.cat([text_emb, fp_expanded], dim=-1)
        gate = self.sigmoid(self.gate_linear(concat_feat))

        fused = text_emb + gate * fp_expanded
        return self.norm(fused)


# Restore original design 2: Text-graph cross-attention alignment gated layer (ZeroInitDeepFusion)
class ZeroInitDeepFusion(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super(ZeroInitDeepFusion, self).__init__()
        # Use Cross-Attention to solve the alignment problem between 1D sequences and 2D graphs
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Also preserve your excellent zero-initialization protection mechanism
        nn.init.constant_(self.gate_linear.bias, -10.0)
        nn.init.xavier_uniform_(self.gate_linear.weight)

    def forward(self, text_emb, graph_emb, graph_padding_mask):
        """
        text_emb: [B, SeqLen, D] (As Query)
        graph_emb: [B, MaxNodes, D] (As Key and Value)
        graph_padding_mask: [B, MaxNodes] (Mask empty nodes)
        """
        # 1. Cross-modal cross-attention addressing: use text to find the corresponding graph structure
        # The length of attn_output will strictly equal SeqLen, i.e., [B, SeqLen, D]
        attn_output, _ = self.cross_attn(
            query=text_emb,
            key=graph_emb,
            value=graph_emb,
            key_padding_mask=graph_padding_mask
        )
        attn_output = self.dropout(attn_output)

        # 2. Gated residual fusion: safely absorb the retrieved graph features into the text
        concat_feat = torch.cat([text_emb, attn_output], dim=-1)
        gate = self.sigmoid(self.gate_linear(concat_feat))

        fused = text_emb + gate * attn_output
        return self.norm(fused)


# ================= 3. Main Model =================
class BioRetroTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, atom_feat_dim, d_model=512, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1,
                 padding_idx=1, max_len=5000, fp_dim=2048):
        super(BioRetroTransformer, self).__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=padding_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True),
            num_encoder_layers)

        self.graph_encoder = GraphEncoder(atom_feat_dim, d_model, num_layers=3, dropout=dropout)

        # Restore cascaded fusion layers
        self.fp_fusion_layer = FingerprintGateFusion(fp_dim, d_model, dropout)
        self.graph_fusion_layer = ZeroInitDeepFusion(d_model, nhead, dropout)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True),
            num_decoder_layers)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, graph_x, graph_adj, fp,
                src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):

        # 1. Independent feature encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory_seq = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        memory_graph = self.graph_encoder(graph_x, graph_adj)
        graph_padding_mask = (graph_x.abs().sum(dim=-1) == 0)

        # 2. Cascaded absorption: sequence absorbs fingerprints first, then uses Cross-Attention to absorb aligned graphs
        memory_seq = self.fp_fusion_layer(memory_seq, fp)
        memory_fused = self.graph_fusion_layer(memory_seq, memory_graph, graph_padding_mask)

        # 3. [Core purification: Directly output single-path fused features]
        # Absolutely do NOT do redundant concatenation like torch.cat([memory_fused, memory_graph], dim=1) anymore!
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        output = self.decoder(tgt_emb,
                              memory=memory_fused,  # <- Very pure aligned features
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)  # Mask only covers the sequence itself

        return self.generator(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode_all(self, src, graph_x, graph_adj, fp, src_key_padding_mask=None):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory_seq = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        memory_graph = self.graph_encoder(graph_x, graph_adj)
        graph_padding_mask = (graph_x.abs().sum(dim=-1) == 0)

        memory_seq = self.fp_fusion_layer(memory_seq, fp)
        memory_fused = self.graph_fusion_layer(memory_seq, memory_graph, graph_padding_mask)

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(memory_seq.shape[:2], dtype=torch.bool, device=src.device)

        return memory_fused, src_key_padding_mask