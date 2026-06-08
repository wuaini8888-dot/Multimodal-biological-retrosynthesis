import torch
import yaml
import argparse
import math
import os
import sys
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# --- Critical fix: Point to the exact absolute path containing graph_utils.py ---
sys.path.append('/data/stu1/ml_project/code/Substrate/single')

from graph_utils import smiles_to_graph, ATOM_FEAT_DIM
from model import BioRetroTransformer
from data_loader import Vocab, get_morgan_fingerprint


class BeamNode:
    def __init__(self, sequence, score, log_prob):
        self.sequence = sequence
        self.score = score
        self.log_prob = log_prob


def predict(model, src_line, src_vocab, tgt_vocab, device, max_nodes=150, beam_width=10, max_len=500, alpha=0.7):
    model.eval()
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

    fp_array = get_morgan_fingerprint(raw_smiles)
    fp_tensor = torch.tensor(fp_array, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        memory, memory_mask = model.encode_all(src_tensor, graph_x, graph_adj, fp_tensor)

    start_token = tgt_vocab.token2id.get('<s>', 2)
    end_token = tgt_vocab.token2id.get('</s>', 3)

    finished_beams = []
    beams = [BeamNode([start_token], 0.0, None)]

    for step in range(max_len):
        candidates = []
        for beam in beams:
            if beam.sequence[-1] == end_token:
                candidates.append(beam)
                continue

            tgt_input = torch.tensor([beam.sequence], dtype=torch.long).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            with torch.no_grad():
                output = model.decoder(
                    model.pos_encoder(model.tgt_embedding(tgt_input) * math.sqrt(model.d_model)),
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_mask
                )
                logits = model.generator(output[:, -1, :])
                probs = torch.log_softmax(logits, dim=-1)

            topk_probs, topk_ids = probs.topk(beam_width)
            for i in range(beam_width):
                token_id = topk_ids[0, i].item()
                score = topk_probs[0, i].item()
                candidates.append(BeamNode(beam.sequence + [token_id], beam.score + score, None))

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
    parser.add_argument('--config', type=str, default='/data/stu1/ml_project/bioec_retro1/FF/Finger3/train_stage2.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='/data/stu1/ml_project/bioec_retro1/FF/Finger3/save/finetune/20260402_step_200000.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.7)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint}...")

    vocab_src = Vocab(config['src_vocab'])
    vocab_tgt = Vocab(config['tgt_vocab'])

    dropout = config.get('dropout', 0.1)
    if isinstance(dropout, list): dropout = dropout[0]

    model = BioRetroTransformer(
        src_vocab_size=len(vocab_src), tgt_vocab_size=len(vocab_tgt),
        atom_feat_dim=ATOM_FEAT_DIM, d_model=config['d_model'],
        nhead=config['nhead'], num_encoder_layers=config['num_layers'],
        num_decoder_layers=config['num_layers'], dropout=dropout,
        fp_dim=2048
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Start Predicting (Beam Size={args.beam_size}, Max Len={args.max_len}, Alpha={args.alpha})...")

    # --- Hardcode the four input files in the code for batch execution ---
    input_files = [
        '/data/stu1/ml_project/code/dataset/natural_data/mentor_testset/category_processed/clean/flavonoids_src.txt',
        '/data/stu1/ml_project/code/dataset/natural_data/mentor_testset/category_processed/clean/phenols_src.txt',
        '/data/stu1/ml_project/code/dataset/natural_data/mentor_testset/category_processed/clean/steroids_src.txt',
        '/data/stu1/ml_project/code/dataset/natural_data/mentor_testset/category_processed/clean/terpenes_src.txt'
    ]

    output_dir = '/data/stu1/ml_project/code/dataset/natural_data/mentor_testset/category_processed/outputs'
    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        base_name = os.path.basename(input_file).replace('_src.txt', '_pred.txt')
        output_file = os.path.join(output_dir, base_name)

        print(f"\n[{base_name}] Processing input: {input_file}")

        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        with open(output_file, 'w') as f_out:
            for line in tqdm(lines, desc=base_name):
                if not line: continue
                preds = predict(model, line, vocab_src, vocab_tgt, device, beam_width=args.beam_size,
                                max_len=args.max_len,
                                alpha=args.alpha)
                f_out.write(f"{line}\t" + "\t".join(preds) + "\n")

        print(f"[{base_name}] Done! Saved to {output_file}")

    print(f"\nAll 4 files processed successfully! Check your outputs in: {output_dir}")