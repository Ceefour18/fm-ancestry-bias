"""H2: fine-tune Geneformer V2-104M for coarse cell-type classification on one regime/seed; predict test
(or EA validation with --val) donors. Randolph mock (NI) cells only."""
import argparse, json, os, time, numpy as np, pandas as pd, torch, anndata
from datasets import load_from_disk
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
p = argparse.ArgumentParser()
p.add_argument("--regime", required=True, choices=["ea_only", "balanced", "random"])
p.add_argument("--seed", type=int, default=2026); p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--epochs", type=float, default=3); p.add_argument("--batch", type=int, default=8)
p.add_argument("--val", action="store_true"); p.add_argument("--max_cells", type=int, default=0)
a = p.parse_args(); t0 = time.time()
CLASSES = ["B", "CD4_T", "CD8_T", "DC", "Monocyte", "NK"]
H = json.load(open("results/h2_splits.json"))
o = anndata.read_h5ad("data/processed/randolph_labelled.h5ad", backed="r").obs
o = o[o.condition == "NI"]; lab = o.label_coarse.astype(str).to_dict(); don = o.subject_id.astype(str).to_dict()
train_d = set(H["regimes"][str(a.seed)][a.regime]); eval_d = set(H["val"] if a.val else H["test"])
ds = load_from_disk("data/processed/geneformer/randolph/tokens.dataset")
ds = ds.filter(lambda b: [c in lab for c in b["cell_id"]], batched=True, num_proc=16)
ds = ds.map(lambda b: {"label": [CLASSES.index(lab[c]) for c in b["cell_id"]],
                       "donor": [don[c] for c in b["cell_id"]]}, batched=True, num_proc=16)
tr = ds.filter(lambda b: [d in train_d for d in b["donor"]], batched=True, num_proc=16)
te = ds.filter(lambda b: [d in eval_d for d in b["donor"]], batched=True, num_proc=16)
if a.max_cells:
    tr = tr.shuffle(seed=a.seed).select(range(min(a.max_cells, len(tr)))); te = te.select(range(min(a.max_cells, len(te))))
print(f"train {len(tr)} cells ({len(train_d)} donors) | eval {len(te)} cells ({len(eval_d)} donors)", flush=True)

def collate(batch):                                   # pad with token id 0 (<pad>)
    L = max(len(x["input_ids"]) for x in batch)
    ids = torch.zeros(len(batch), L, dtype=torch.long); att = torch.zeros(len(batch), L, dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"]); ids[i, :n] = torch.tensor(x["input_ids"]); att[i, :n] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": torch.tensor([x["label"] for x in batch])}

model = BertForSequenceClassification.from_pretrained("models/Geneformer/Geneformer-V2-104M", num_labels=len(CLASSES))
tag = f"{a.regime}_s{a.seed}_lr{a.lr:g}" + ("_val" if a.val else "") + (f"_test{a.max_cells}" if a.max_cells else "")
args = TrainingArguments(output_dir=f"checkpoints/h2/geneformer_{tag}", learning_rate=a.lr, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=2 * a.batch, warmup_ratio=0.1,
        weight_decay=0.001, lr_scheduler_type="linear", bf16=True, logging_steps=200, save_strategy="no",
        report_to=[], seed=a.seed, dataloader_num_workers=4, remove_unused_columns=False)
trainer = Trainer(model=model, args=args, train_dataset=tr, data_collator=collate)
trainer.train()
pred = trainer.predict(te).predictions.argmax(1)
os.makedirs("results/preds_h2", exist_ok=True)
out = pd.DataFrame({"cell_id": te["cell_id"], "subject_id": te["donor"],
                    "label": [CLASSES[i] for i in te["label"]], "pred_coarse": [CLASSES[i] for i in pred]})
out.to_csv(f"results/preds_h2/geneformer_{tag}.csv.gz", index=False)
print(f"pooled macro-F1 {f1_score(out.label, out.pred_coarse, average='macro'):.3f} | {round(time.time() - t0)} s", flush=True)