import torch
import yaml
import argparse
import math
import os
import sys
from tqdm import tqdm

# ==========================================
# Core: Automatically import your original directory to reuse model and data processing code
# ==========================================
sys.path.append("/data/stu1/ml_project/bioec_retro1/FF/Finger3")

from graph_utils import smiles_to_graph, ATOM_FEAT_DIM
from model import BioRetroTransformer
from data_loader import Vocab, get_morgan_fingerprint

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class BeamNode:
    def __init__(self, sequence, score, log_prob):
        self.sequence = sequence
        self.score = score
        self.log_prob = log_prob


def predict_ensemble(models, src_line, src_vocab, tgt_vocab, device, max_nodes=150, beam_width=10, max_len=500,
                     alpha=0.7):
    for m in models:
        m.eval()

    src_tokens = src_line.strip().split()
    src_ids = src_vocab.encode(src_tokens)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    raw_smiles = "".join(src_tokens)

    try:
        graph_x, graph_adj = smiles_to_graph(raw_smiles, max_nodes)
    except:
        import numpy as np
        graph_x = np.zeros((max_nodes, ATOM_FEAT_DIM))
        graph_adj = np.zeros((4, max_nodes, max_nodes))
        graph_x[0, :] = 1.0
        for ch in range(4): graph_adj[ch, 0, 0] = 1.0

    graph_x = torch.tensor(graph_x, dtype=torch.float).unsqueeze(0).to(device)
    graph_adj = torch.tensor(graph_adj, dtype=torch.float).unsqueeze(0).to(device)
    fp_tensor = torch.tensor(get_morgan_fingerprint(raw_smiles), dtype=torch.float).unsqueeze(0).to(device)

    # 1. Independent encoding: Obtain the respective Memory feature spaces of the 4 models
    memories = []
    memory_masks = []
    with torch.no_grad():
        for m in models:
            mem, mask = m.encode_all(src_tensor, graph_x, graph_adj, fp_tensor)
            memories.append(mem)
            memory_masks.append(mask)

    start_token = tgt_vocab.token2id.get('<s>', 2)
    end_token = tgt_vocab.token2id.get('</s>', 3)
    beams = [BeamNode([start_token], 0.0, None)]

    for step in range(max_len):
        candidates = []
        for beam in beams:
            if beam.sequence[-1] == end_token:
                candidates.append(beam)
                continue

            tgt_input = torch.tensor([beam.sequence], dtype=torch.long).to(device)
            tgt_mask = models[0].generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            # 2. Logit fusion: Average the 4 models in the Softmax probability space
            avg_probs = 0
            with torch.no_grad():
                for i, m in enumerate(models):
                    output = m.decoder(
                        m.pos_encoder(m.tgt_embedding(tgt_input) * math.sqrt(m.d_model)),
                        memories[i],
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=memory_masks[i]
                    )
                    logits = m.generator(output[:, -1, :])
                    avg_probs += torch.softmax(logits, dim=-1)

                avg_probs = avg_probs / len(models)
                probs = torch.log(avg_probs + 1e-9)

            topk_probs, topk_ids = probs.topk(beam_width)
            for i in range(beam_width):
                token_id = topk_ids[0, i].item()
                score = topk_probs[0, i].item()
                candidates.append(BeamNode(beam.sequence + [token_id], beam.score + score, None))

        # 3. Real-time RDKit filtering mechanism
        def get_normalized_score(node):
            length = len(node.sequence)
            penalty = math.pow((5 + length) / 6, alpha)
            score = node.score / penalty

            if node.sequence[-1] == end_token:
                seq = node.sequence
                if seq[0] == start_token: seq = seq[1:]
                if seq and seq[-1] == end_token: seq = seq[:-1]
                smi = "".join(tgt_vocab.decode(seq))
                if Chem.MolFromSmiles(smi) is None:
                    score -= 1000.0
            return score

        candidates.sort(key=lambda x: get_normalized_score(x), reverse=True)
        beams = candidates[:beam_width]

        if all(b.sequence[-1] == end_token for b in beams):
            break

    all_preds = []
    for beam in beams:
        seq = beam.sequence
        if seq[0] == start_token: seq = seq[1:]
        if seq and seq[-1] == end_token: seq = seq[:-1]
        all_preds.append(" ".join(tgt_vocab.decode(seq)))
    return all_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help='Pass the .pt paths of the 4 models')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.7)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    vocab_src = Vocab(config['src_vocab'])
    vocab_tgt = Vocab(config['tgt_vocab'])
    dropout = config.get('dropout', 0.1)
    if isinstance(dropout, list): dropout = dropout[0]

    print(f"Loading {len(args.checkpoints)} ensemble sub-models in parallel...")
    ensemble_models = []
    for idx, ckpt_path in enumerate(args.checkpoints):
        model = BioRetroTransformer(
            src_vocab_size=len(vocab_src), tgt_vocab_size=len(vocab_tgt),
            atom_feat_dim=ATOM_FEAT_DIM, d_model=config['d_model'],
            nhead=config['nhead'], num_encoder_layers=config['num_layers'],
            num_decoder_layers=config['num_layers'], dropout=dropout, fp_dim=2048
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        ensemble_models.append(model)
        print(f"Model {idx + 1} loaded successfully: {ckpt_path}")

    print("Starting joint decision testing for the four models...")
    with open(args.input, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    with open(args.output, 'w') as f_out:
        for line in tqdm(lines):
            if not line: continue
            preds = predict_ensemble(ensemble_models, line, vocab_src, vocab_tgt, device,
                                     beam_width=args.beam_size, max_len=args.max_len, alpha=args.alpha)
            f_out.write(f"{line}\t" + "\t".join(preds) + "\n")
            f_out.flush()

    print(f"Ensemble inference complete! Results saved to: {args.output}")