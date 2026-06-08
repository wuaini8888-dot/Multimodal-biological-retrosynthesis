import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import BioRetroTransformer
from data_loader import Vocab, build_weighted_loader
from graph_utils import ATOM_FEAT_DIM
from utils import set_seed, NoamOpt

def get_latest_checkpoint(save_prefix):
    save_dir = os.path.dirname(save_prefix)
    base_name = os.path.basename(save_prefix)
    if not os.path.exists(save_dir): return None, 0

    ckpts = [f for f in os.listdir(save_dir) if f.startswith(base_name) and f.endswith('.pt')]
    if not ckpts: return None, 0

    ckpts.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)) if re.search(r'step_(\d+)', x) else 0)
    latest_ckpt = os.path.join(save_dir, ckpts[-1])
    latest_step = int(re.search(r'step_(\d+)', latest_ckpt).group(1))
    return latest_ckpt, latest_step


def train(config, resume_checkpoint=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    set_seed(config.get('seed', 2026))

    accum_count = config.get('accum_count', 1)
    if isinstance(accum_count, list): accum_count = accum_count[0]
    config['accum_count'] = accum_count

    dropout = config.get('dropout', 0.1)
    if isinstance(dropout, list): dropout = dropout[0]

    try:
        src_vocab_path = config['src_vocab']
        tgt_vocab_path = config['tgt_vocab']
        uspto_src = config['data']['corpus_1']['path_src']
        uspto_tgt = config['data']['corpus_1']['path_tgt']
        bio_src = config['data']['corpus_2']['path_src']
        bio_tgt = config['data']['corpus_2']['path_tgt']

        uspto_weight = float(config['data']['corpus_1'].get('weight', 1.0))
        bio_weight = float(config['data']['corpus_2'].get('weight', 1.0))
    except KeyError as e:
        raise KeyError(f"Configuration file parsing error: {e}")

    vocab_src = Vocab(src_vocab_path)
    vocab_tgt = Vocab(tgt_vocab_path)
    print(f"Vocab Size: {len(vocab_src)} / {len(vocab_tgt)}")

    loader_config = {
        'uspto_src': uspto_src, 'uspto_tgt': uspto_tgt,
        'bio_src': bio_src, 'bio_tgt': bio_tgt,
        'uspto_weight': uspto_weight,
        'bio_weight': bio_weight,
        'batch_size': config['batch_size'],
        'max_graph_nodes': config.get('max_graph_nodes', 150)
    }
    train_loader = build_weighted_loader(vocab_src, vocab_tgt, loader_config)

    # New architecture model including Gated FP
    model = BioRetroTransformer(
        src_vocab_size=len(vocab_src), tgt_vocab_size=len(vocab_tgt),
        atom_feat_dim=ATOM_FEAT_DIM, d_model=config['d_model'],
        nhead=config['nhead'], num_encoder_layers=config['num_layers'],
        num_decoder_layers=config['num_layers'], dropout=dropout,
        fp_dim=2048 # Specify fingerprint dimension
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    opt_scheduler = NoamOpt(config['d_model'], 2, 8000, optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1)

    start_step = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f" Resuming from: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        try:
            step_match = re.search(r'step_(\d+)', resume_checkpoint)
            if step_match:
                start_step = int(step_match.group(1))
                opt_scheduler._step = start_step
                print(f" Restored step: {start_step}")
        except:
            print(" Warning: Could not parse step from filename.")
    else:
        print(" Starting training from scratch (Gated FP + R-GCN Baseline)...")

    model.train()

    save_dir = os.path.dirname(config['save_model'])
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    global_step = start_step
    total_loss = 0
    optimizer.zero_grad()

    total_steps = config.get('train_steps', 200000)
    steps_per_epoch = len(train_loader) // accum_count
    if steps_per_epoch == 0: steps_per_epoch = 1
    start_epoch = global_step // steps_per_epoch
    num_epochs = (total_steps // steps_per_epoch) + 1

    last_saved_path = resume_checkpoint

    for epoch in range(start_epoch, num_epochs):
        for i, batch in enumerate(train_loader):
            # Unpack including fp
            src, tgt, graph_x, graph_adj, fp = [b.to(device) for b in batch]

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            src_key_padding_mask = (src == 1)
            tgt_key_padding_mask = (tgt_input == 1)

            # Forward pass including fp
            output = model(src, tgt_input, graph_x, graph_adj, fp=fp,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           tgt_mask=tgt_mask)

            loss = criterion(output.reshape(-1, len(vocab_tgt)), tgt_output.reshape(-1))
            loss = loss / accum_count
            loss.backward()

            total_loss += loss.item() * accum_count

            if (i + 1) % accum_count == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_scheduler.step()
                opt_scheduler.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg = total_loss / (50 * accum_count)
                    cur_lr = opt_scheduler._rate
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {avg:.4f} | LR: {cur_lr:.7f}")
                    total_loss = 0

                if global_step % config.get('save_checkpoint_steps', 2000) == 0:
                    save_path = f"{config['save_model']}_step_{global_step}.pt"
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved: {save_path}")
                    last_saved_path = save_path

            if global_step >= total_steps:
                if global_step % config.get('save_checkpoint_steps', 2000) != 0:
                    save_path = f"{config['save_model']}_step_{global_step}.pt"
                    torch.save(model.state_dict(), save_path)
                return last_saved_path

    return last_saved_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config1', type=str, default='/data/stu1/ml_project/code/Substrate/single/train_stage1.yaml')
    parser.add_argument('--config2', type=str, default='/data/stu1/ml_project/code/Substrate/single/train_stage2.yaml')
    args = parser.parse_args()

    with open(args.config1, 'r') as f:
        config1 = yaml.safe_load(f)
    with open(args.config2, 'r') as f:
        config2 = yaml.safe_load(f)

    # ... Calling logic for stages 1 and 2 remains completely unchanged ...
    print("\n" + "=" * 50)
    print(" STAGE 1: PRE-TRAINING (USPTO Data Only)")
    print("=" * 50)
    stage1_prefix = config1['save_model']
    latest_ckpt1, latest_step1 = get_latest_checkpoint(stage1_prefix)
    stage1_target_steps = config1.get('train_steps', 150000)

    if latest_step1 >= stage1_target_steps:
        final_stage1_ckpt = latest_ckpt1   # If 150,000 steps are already reached, do not execute train()
    else:
        final_stage1_ckpt = train(config1, resume_checkpoint=latest_ckpt1)

    print("\n" + "=" * 50)
    print(" STAGE 2: FINE-TUNING (Bio Data Only)")
    print("=" * 50)
    stage2_prefix = config2['save_model']
    latest_ckpt2, latest_step2 = get_latest_checkpoint(stage2_prefix)
    stage2_target_steps = config2.get('train_steps', 200000)    # Triggers training only if current steps < 200,000

    if latest_step2 < stage2_target_steps:
        resume_ckpt = latest_ckpt2 if latest_ckpt2 else final_stage1_ckpt
        train(config2, resume_checkpoint=resume_ckpt)