"""H2: fine-tune scGPT (whole-human) for coarse cell-type classification on one regime/seed; predict test
(or EA validation with --val) donors. Randolph mock (NI) cells only. No flash-attention (weights remapped)."""
import argparse, json, os, time, random, numpy as np, pandas as pd, torch, anndata
from torch import nn
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.preprocess import Preprocessor
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from sklearn.metrics import f1_score
p = argparse.ArgumentParser()
p.add_argument("--regime", required=True, choices=["ea_only", "balanced", "random"])
p.add_argument("--seed", type=int, default=2026); p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch", type=int, default=32)
p.add_argument("--val", action="store_true"); p.add_argument("--max_cells", type=int, default=0)
a = p.parse_args(); t0 = time.time()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
CLASSES = ["B", "CD4_T", "CD8_T", "DC", "Monocyte", "NK"]; MD = "models/scGPT_human"; MAXLEN = 1201
H = json.load(open("results/h2_splits.json"))
obs = pd.read_csv("data/processed/randolph_obs.csv.gz", index_col=0)
obs = obs[obs.condition == "NI"]
train_d = set(H["regimes"][str(a.seed)][a.regime]); eval_d = set(H["val"] if a.val else H["test"])

vocab = GeneVocab.from_file(f"{MD}/vocab.json")
for s in ["<pad>", "<cls>", "<eoc>"]:
    if s not in vocab: vocab.append_token(s)
vocab.set_default_index(vocab["<pad>"])
ad = anndata.read_h5ad("data/processed/uce/randolph/cells.h5ad")          # raw counts, gene symbols
ad = ad[ad.obs_names.isin(obs.index)].copy()
ad.obs["donor"] = obs.loc[ad.obs_names, "subject_id"].astype(str).values
ad.obs["label"] = obs.loc[ad.obs_names, "label_coarse"].astype(str).values
ad = ad[ad.obs.donor.isin(train_d | eval_d)].copy()
ad = ad[:, [g in vocab for g in ad.var_names]].copy()
Preprocessor(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False, normalize_total=1e4,
             result_normed_key="X_normed", log1p=True, result_log1p_key="X_log1p", subset_hvg=False,
             binning=51, result_binned_key="X_binned")(ad, batch_key=None)
gene_ids = np.array(vocab(ad.var_names.tolist()), dtype=int)

def tokens(sub):
    X = sub.layers["X_binned"]; X = X.toarray() if hasattr(X, "toarray") else X
    t = tokenize_and_pad_batch(X, gene_ids, max_len=MAXLEN, vocab=vocab, pad_token="<pad>", pad_value=-2,
                               append_cls=True, include_zero_gene=False)
    return t["genes"], t["values"].float()
tr = ad[ad.obs.donor.isin(train_d)]; te = ad[ad.obs.donor.isin(eval_d)]
if a.max_cells:
    tr = tr[np.random.default_rng(a.seed).choice(tr.n_obs, min(a.max_cells, tr.n_obs), replace=False)]
    te = te[:min(a.max_cells, te.n_obs)]
gtr, vtr = tokens(tr); gte, vte = tokens(te)
ytr = torch.tensor([CLASSES.index(x) for x in tr.obs.label])
print(f"train {tr.n_obs} cells ({len(train_d)} donors) | eval {te.n_obs} cells ({len(eval_d)} donors) | {len(gene_ids)} genes", flush=True)

args = json.load(open(f"{MD}/args.json"))
model = TransformerModel(len(vocab), args["embsize"], args["nheads"], args["d_hid"], args["nlayers"],
        nlayers_cls=3, n_cls=len(CLASSES), vocab=vocab, dropout=0.2, pad_token="<pad>", pad_value=-2,
        do_mvc=False, do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
        input_emb_style="continuous", n_input_bins=51, cell_emb_style="cls", explicit_zero_prob=False,
        use_fast_transformer=False, pre_norm=False)
before = {k: v.clone() for k, v in model.state_dict().items()}
load_pretrained(model, torch.load(f"{MD}/best_model.pt", map_location="cpu"), verbose=False)
changed = sum(not torch.equal(before[k], v) for k, v in model.state_dict().items())
print(f"pretrained tensors loaded: {changed} of {len(before)} (the rest = new classifier head)", flush=True)
dev = torch.device("cuda"); model.to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-4)
sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9); scaler = torch.cuda.amp.GradScaler(); ce = nn.CrossEntropyLoss()
pad = vocab["<pad>"]
for ep in range(a.epochs):
    model.train(); perm = torch.randperm(len(ytr)); tot = 0
    for i in range(0, len(perm), a.batch):
        j = perm[i:i + a.batch]; g, v, y = gtr[j].to(dev), vtr[j].to(dev), ytr[j].to(dev)
        with torch.cuda.amp.autocast():
            out = model(g, v, src_key_padding_mask=g.eq(pad), batch_labels=None, CLS=True, CCE=False, MVC=False, ECS=False)
            loss = ce(out["cls_output"], y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); tot += loss.item() * len(j)
    sched.step(); print(f"epoch {ep + 1}: loss {tot / len(ytr):.4f} | {round(time.time() - t0)} s", flush=True)
model.eval(); pred = []
with torch.no_grad(), torch.cuda.amp.autocast():
    for i in range(0, len(gte), 2 * a.batch):
        g, v = gte[i:i + 2 * a.batch].to(dev), vte[i:i + 2 * a.batch].to(dev)
        pred.append(model(g, v, src_key_padding_mask=g.eq(pad), batch_labels=None, CLS=True, CCE=False,
                          MVC=False, ECS=False)["cls_output"].argmax(1).cpu())
pred = torch.cat(pred).numpy()
tag = f"{a.regime}_s{a.seed}_lr{a.lr:g}" + ("_val" if a.val else "") + (f"_test{a.max_cells}" if a.max_cells else "")
os.makedirs("results/preds_h2", exist_ok=True)
out = pd.DataFrame({"cell_id": te.obs_names, "subject_id": te.obs.donor.values, "label": te.obs.label.values,
                    "pred_coarse": [CLASSES[i] for i in pred]})
out.to_csv(f"results/preds_h2/scgpt_{tag}.csv.gz", index=False)
print(f"pooled macro-F1 {f1_score(out.label, out.pred_coarse, average='macro'):.3f} | {round(time.time() - t0)} s", flush=True)