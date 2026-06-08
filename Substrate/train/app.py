import os
import sys
import yaml
import torch
import pickle
import base64
import io
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

# Block RDKit warnings
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Path and environment configuration (Ultimate memory isolation method)
# ==========================================
sys.path.insert(0, "/data/stu1/ml_project/bioec_retro1/FF/Finger3")

from data_loader import Vocab
from predict_ensemble import predict_ensemble, BioRetroTransformer

if 'graph_utils' in sys.modules:
    del sys.modules['graph_utils']

sys.path.insert(0, "/data/stu1/ml_project/bioec_retro1/EC")

from ec_model import BioEC_EnzymePredictor, BioDualModalEncoder
from ec_data import HierarchicalECDataset

# ==========================================
# 2. Global parameters and model loading
# ==========================================
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Starting system, using device: {DEVICE}")

ENSEMBLE_YAML = '/data/stu1/ml_project/bioec_retro1/FF/Finger3/train_stage2.yaml'
ENSEMBLE_CKPTS = [
    '/data/stu1/ml_project/bioec_retro1/FF/Finger3/save/finetune/20260402_step_200000.pt',
    '/data/stu1/ml_project/bioec_retro1/train1/save2/finetune_seed2/model_step_200000.pt',
    '/data/stu1/ml_project/bioec_retro1/train1/save3/finetune_seed3/model_step_200000.pt',
    '/data/stu1/ml_project/bioec_retro1/train1/save4/finetune_seed4/model_step_200000.pt'
]
EC_PRETRAINED_PATH = '/data/stu1/ml_project/bioec_retro1/EC/deepchem_bert'
EC_CKPT = '/data/stu1/ml_project/bioec_retro1/EC/save_ec_siamese_optimized/best_model.pt'
EC_LABEL_MAPS = '/data/stu1/ml_project/bioec_retro1/EC/save_ec_siamese_optimized/label_maps.pkl'
EC_TEST_CSV = '/data/stu1/ml_project/bioec_retro1/dataset/EC/clean_processed_csv/test.csv'

# ---------- Load Ensemble Model (Stage A) ----------
with open(ENSEMBLE_YAML, 'r') as f:
    ens_config = yaml.safe_load(f)

vocab_src = Vocab(ens_config['src_vocab'])
vocab_tgt = Vocab(ens_config['tgt_vocab'])
dropout = ens_config.get('dropout', 0.1)
if isinstance(dropout, list): dropout = dropout[0]

ensemble_models = []
for idx, ckpt_path in enumerate(ENSEMBLE_CKPTS):
    model = BioRetroTransformer(
        src_vocab_size=len(vocab_src), tgt_vocab_size=len(vocab_tgt),
        atom_feat_dim=70, d_model=ens_config['d_model'],
        nhead=ens_config['nhead'], num_encoder_layers=ens_config['num_layers'],
        num_decoder_layers=ens_config['num_layers'], dropout=dropout, fp_dim=2048
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    ensemble_models.append(model)

# ---------- Load EC Model (Stage B) ----------
tokenizer = AutoTokenizer.from_pretrained(EC_PRETRAINED_PATH)
with open(EC_LABEL_MAPS, 'rb') as f:
    label_maps = pickle.load(f)

num_classes_list = [len(m) for m in label_maps]
inv_label_maps = [{v: k for k, v in m.items()} for m in label_maps]

encoder = BioDualModalEncoder(EC_PRETRAINED_PATH, atom_feat_dim=67)
ec_model = BioEC_EnzymePredictor(encoder, num_classes_list, hidden_dim=512)
ec_model.load_state_dict(torch.load(EC_CKPT, map_location=DEVICE))
ec_model.to(DEVICE)
ec_model.eval()

dummy_dataset = HierarchicalECDataset(EC_TEST_CSV, tokenizer, max_len=512, label_maps=label_maps, is_train=False)


# ==========================================
# 3. Core tools and inference functions
# ==========================================
def canonicalize_smiles(smi):
    if not smi or not isinstance(smi, str): return None
    try:
        smi = smi.replace(" ", "")
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except:
        return None


def draw_smiles_to_html(smiles, width=250, height=250):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return "Invalid chemical structure"
    img = Draw.MolToImage(mol, size=(width, height))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_str}" width="{width}" height="{height}" style="margin:auto;" />'


def beam_search_ec(model, inputs_R_dict, inputs_P_dict, inv_maps, top_k):
    """Cascade Beam Search specially designed for the cascaded neuro-symbolic system"""
    with torch.no_grad():
        # Extract tri-modal features of substrate and product
        feat_R, seq_R, mask_R = model.encoder(
            inputs_R_dict['input_ids'], inputs_R_dict['attention_mask'], inputs_R_dict['graph_x'],
            inputs_R_dict['graph_adj'], inputs_R_dict['fp_indices'], inputs_R_dict['fp_mask'])

        feat_P, seq_P, mask_P = model.encoder(
            inputs_P_dict['input_ids'], inputs_P_dict['attention_mask'], inputs_P_dict['graph_x'],
            inputs_P_dict['graph_adj'], inputs_P_dict['fp_indices'], inputs_P_dict['fp_mask'])

        # Topology and fingerprint alignment difference
        aligned_R, _ = model.cross_attn(query=seq_R, key=seq_P, value=seq_P, key_padding_mask=mask_P)
        aligned_R = model.norm_R(seq_R + aligned_R)
        aligned_P, _ = model.cross_attn(query=seq_P, key=seq_R, value=seq_R, key_padding_mask=mask_R)
        aligned_P = model.norm_P(seq_P + aligned_P)

        delta_R = model.delta_pooling(aligned_R - seq_R, mask_R)
        delta_P = model.delta_pooling(aligned_P - seq_P, mask_P)

        fp_delta_feat = model.fp_delta_proj(inputs_P_dict['fp_dense'] - inputs_R_dict['fp_dense'])

        # Core fusion features
        features = model.reaction_proj(torch.cat([feat_R, feat_P, delta_R, delta_P, fp_delta_feat], dim=-1))

        # Level 1 (Main class) prediction
        probs1 = torch.nn.functional.softmax(model.head1(features), dim=-1)[0]
        topk_p1, topk_i1 = torch.topk(probs1, min(top_k, probs1.size(0)))
        beams = [{'prob': p.item(), 'path': [i.item()]} for p, i in zip(topk_p1, topk_i1)]

        # Level 2 (Subclass) prediction
        new_beams = []
        for b in beams:
            pred1 = torch.tensor([b['path'][0]], device=DEVICE)
            feat_l2 = model.dropout(model.relu(torch.cat([features, model.emb1(pred1)], dim=-1)))
            probs2 = torch.nn.functional.softmax(model.head2(feat_l2), dim=-1)[0]
            topk_p2, topk_i2 = torch.topk(probs2, min(top_k, probs2.size(0)))
            for p, i in zip(topk_p2, topk_i2):
                new_beams.append({'prob': b['prob'] * p.item(), 'path': b['path'] + [i.item()]})
        beams = sorted(new_beams, key=lambda x: x['prob'], reverse=True)[:top_k]

        # Level 3 (Sub-subclass) prediction
        new_beams = []
        for b in beams:
            pred2 = torch.tensor([b['path'][1]], device=DEVICE)
            feat_l3 = model.dropout(model.relu(torch.cat([features, model.emb2(pred2)], dim=-1)))
            probs3 = torch.nn.functional.softmax(model.head3(feat_l3), dim=-1)[0]
            topk_p3, topk_i3 = torch.topk(probs3, min(top_k, probs3.size(0)))
            for p, i in zip(topk_p3, topk_i3):
                new_beams.append({'prob': b['prob'] * p.item(), 'path': b['path'] + [i.item()]})
        beams = sorted(new_beams, key=lambda x: x['prob'], reverse=True)[:top_k]

        # Level 4 (Serial number) prediction
        new_beams = []
        for b in beams:
            pred3 = torch.tensor([b['path'][2]], device=DEVICE)
            feat_l4 = model.dropout(model.relu(torch.cat([features, model.emb3(pred3)], dim=-1)))
            probs4 = torch.nn.functional.softmax(model.head4(feat_l4), dim=-1)[0]
            topk_p4, topk_i4 = torch.topk(probs4, min(top_k, probs4.size(0)))
            for p, i in zip(topk_p4, topk_i4):
                new_beams.append({'prob': b['prob'] * p.item(), 'path': b['path'] + [i.item()]})
        beams = sorted(new_beams, key=lambda x: x['prob'], reverse=True)[:top_k]

        # Format output
        results_str = []
        for idx, b in enumerate(beams):
            ec = f"{inv_maps[0].get(b['path'][0], '0')}.{inv_maps[1].get(b['path'][1], '0')}.{inv_maps[2].get(b['path'][2], '0')}.{inv_maps[3].get(b['path'][3], '0')}"
            prob_percent = b['prob'] * 100
            # Use Markdown to bold the EC number
            results_str.append(
                f"{idx + 1}. **{ec}** ")

        return "<br>".join(results_str)


def run_retrosynthesis(product_smiles, beam_size, ec_topk):
    product_clean = canonicalize_smiles(product_smiles)
    if not product_clean:
        return pd.DataFrame([{"System Prompt": "Invalid product SMILES input, please check."}])

    src_tokens = list(product_clean)
    src_line = " ".join(src_tokens)

    candidates_raw = predict_ensemble(
        models=ensemble_models, src_line=src_line, src_vocab=vocab_src, tgt_vocab=vocab_tgt,
        device=DEVICE, max_nodes=150, beam_width=int(beam_size), max_len=500, alpha=0.7
    )

    results = []

    inputs_P_dict = {}
    inputs_P_dict.update(
        tokenizer(product_clean, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(DEVICE))
    x_P, adj_P = dummy_dataset.smiles_to_graph(product_clean)
    inputs_P_dict['graph_x'] = x_P.unsqueeze(0).to(DEVICE)
    inputs_P_dict['graph_adj'] = adj_P.unsqueeze(0).to(DEVICE)
    fp_dense_P, fp_indices_P = dummy_dataset.smiles_to_fp(product_clean)
    inputs_P_dict['fp_dense'] = fp_dense_P.unsqueeze(0).to(DEVICE)
    inputs_P_dict['fp_indices'] = fp_indices_P.unsqueeze(0).to(DEVICE)
    inputs_P_dict['fp_mask'] = torch.zeros(fp_indices_P.size(0), dtype=torch.bool).unsqueeze(0).to(DEVICE)

    for rank, cand_smi in enumerate(candidates_raw):
        cand_clean = canonicalize_smiles(cand_smi.replace(" ", ""))
        if not cand_clean: continue

        inputs_R_dict = {}
        inputs_R_dict.update(
            tokenizer(cand_clean, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(
                DEVICE))
        x_R, adj_R = dummy_dataset.smiles_to_graph(cand_clean)
        inputs_R_dict['graph_x'] = x_R.unsqueeze(0).to(DEVICE)
        inputs_R_dict['graph_adj'] = adj_R.unsqueeze(0).to(DEVICE)
        fp_dense_R, fp_indices_R = dummy_dataset.smiles_to_fp(cand_clean)
        inputs_R_dict['fp_dense'] = fp_dense_R.unsqueeze(0).to(DEVICE)
        inputs_R_dict['fp_indices'] = fp_indices_R.unsqueeze(0).to(DEVICE)
        inputs_R_dict['fp_mask'] = torch.zeros(fp_indices_R.size(0), dtype=torch.bool).unsqueeze(0).to(DEVICE)

        # Call the dedicated Top-K beam search algorithm
        ec_predictions_html = beam_search_ec(ec_model, inputs_R_dict, inputs_P_dict, inv_label_maps, int(ec_topk))

        mol_img_html = draw_smiles_to_html(cand_clean)

        results.append({
            "Priority Rank": f"Top-{rank + 1}",
            "Candidate Substrate Structure": mol_img_html,
            "Substrate SMILES": cand_clean,
            "Predicted Candidate Enzyme": ec_predictions_html
        })

    if not results:
        return pd.DataFrame([{"System Prompt": "No valid retrosynthetic reactants found"}])

    return pd.DataFrame(results)


# ==========================================
# 4. Gradio Frontend UI Layout
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'> Multi-Modal Biological Retrosynthesis Single-Step Prediction Engine</h1>")
    # gr.Markdown("<p style='text-align: center; color: gray;'>Integrated Logit Probability Fusion Generation Architecture & Cascaded Neuro-Symbolic EC Beam Search Prediction</p>")

    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### Input and Inference Parameter Settings")
            input_smiles = gr.Textbox(
                label="Target Product SMILES",
                placeholder="Example: CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
                lines=3
            )
            beam_slider = gr.Slider(
                minimum=1, maximum=20, value=5, step=1,
                label="Substrate Beam Size",
                info="Number of substrate candidates."
            )
            ec_topk_slider = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="Enzyme Top-K (Number of EC numbers to display)",
                info="How many potential catalytic enzyme options to recommend per substrate."
            )
            submit_btn = gr.Button(" Launch End-to-End Multi-Dimensional Inference", variant="primary", size="lg")

        with gr.Column(scale=3):
            gr.Markdown("### Inference Results Report")
            output_df = gr.Dataframe(
                headers=["Priority Rank", "Candidate Substrate Structure", "Substrate SMILES", "Predicted Candidate Enzyme"],
                datatype=["str", "markdown", "str", "markdown"],  # Enable markdown to parse line breaks and highlights
                interactive=False,
                wrap=True,
                row_count=5
            )

    submit_btn.click(
        fn=run_retrosynthesis,
        inputs=[input_smiles, beam_slider, ec_topk_slider],
        outputs=output_df,
        api_name="predict"
    )

if __name__ == "__main__":
    print("UI service starting... Preparing local port mapping...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)