import numpy as np, pandas as pd, anndata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
t = "data/processed/emb_uce_randolph_test5000_L33_s"
A, B = np.load(t + "2026.npy"), np.load(t + "2027.npy")
cells = pd.read_csv(t + "2026_cells.csv").iloc[:, 0].values
assert (cells == pd.read_csv(t + "2027_cells.csv").iloc[:, 0].values).all()
y = anndata.read_h5ad("data/processed/randolph_labelled.h5ad", backed="r").obs.loc[cells, "label_coarse"].astype(str).values
cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
cv = StratifiedKFold(5, shuffle=True, random_state=2026)
pa = cross_val_predict(KNeighborsClassifier(15), A, y, cv=cv)
pb = cross_val_predict(KNeighborsClassifier(15), B, y, cv=cv)
agree = (pa == pb).mean()
print(f"identical: {np.allclose(A, B)} | cosine median {np.median(cos):.4f}, min {cos.min():.4f}")
print(f"kNN label agreement between seeds: {agree:.4f} -> {'PASS' if agree >= 0.95 else 'FAIL'} (threshold 0.95)")
print(f"kNN accuracy vs labels: seed2026 {(pa == y).mean():.3f}, seed2027 {(pb == y).mean():.3f}")
