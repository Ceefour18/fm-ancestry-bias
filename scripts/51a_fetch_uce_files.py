import hashlib, os, shutil
from huggingface_hub import HfApi, hf_hub_download
DEST = "models/UCE/model_files"
# figshare file: name -> (bytes, md5) from api.figshare.com/v2/articles/24320806
FIG = {"species_offsets.pkl": (139, "ddf9143508857e62607f323b338ff383"),
       "species_chrom.csv": (4097781, "42a9b5871b1ed955fd05414b3ab89081"),
       "4layer_model.torch": (3403514339, "3e2f59d6da6eaa5396aa297edfcec08f"),
       "all_tokens.torch": (2979205876, "3523ea431c403f95701b6eeb0c30b997"),
       "protein_embeddings.tar.gz": (2735410523, "6756d7fa8348e0f70b5e5b11e324f023"),
       "33l_8ep_1024t_1280.torch": (5686283829, "eed62d4fe51873c87bcf9660faa07707")}

api = HfApi()
by_size = {}
for repo in ["minwoosun/uce-misc", "minwoosun/uce-650m"]:
    info = api.model_info(repo, files_metadata=True)
    print("==", repo, "commit", info.sha, flush=True)
    for s in info.siblings:
        print(f"   {s.rfilename:45s} {s.size}", flush=True)
        by_size.setdefault(s.size, (repo, info.sha, s.rfilename))

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

log = open("results/UCE_FILE_PROVENANCE.txt", "w")
for name, (size, want) in FIG.items():
    out = f"{DEST}/{name}"
    if os.path.exists(out) and md5(out) == want:
        print("OK (already)", name, flush=True)
        continue
    if size not in by_size:
        print("MISSING on mirrors:", name, flush=True)
        continue
    repo, sha, fn = by_size[size]
    p = hf_hub_download(repo, fn, revision=sha)
    got = md5(p)
    if got != want:
        print("MD5 MISMATCH:", name, "from", repo, fn, flush=True)
        continue
    shutil.copy(p, out)
    print("OK", name, "<-", repo, fn, flush=True)
    log.write(f"{name}\t{repo}@{sha}/{fn}\tmd5 {got} (matches figshare)\n")
    log.flush()
log.close()
print("FINISHED", flush=True)