# ============================================================
# GENOSCOPE - PARTIE 1/3 : Imports, I/O, QC
# ============================================================
import os, io, json, time, zipfile, warnings, gzip
from io import BytesIO, StringIO
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import requests

warnings.filterwarnings("ignore")

try:
    from Bio import Align
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def cache_data(func):
    return st.cache_data(show_spinner=False)(func)

def _chk(x):
    x = str(x).replace("chr","").replace("Chr","")
    try: return (int(x), int(x))
    except ValueError: return (100, x)

# ------------------------------------------------------------
# I/O : VCF / PLINK / PED / CSV / GFF
# ------------------------------------------------------------
def _read_any(f):
    """Read streamlit file or path to string."""
    if hasattr(f, "read"):
        raw = f.read()
    else:
        with open(f, "rb") as fh: raw = fh.read()
    if isinstance(raw, bytes):
        try: raw = gzip.decompress(raw)
        except Exception: pass
        raw = raw.decode("utf-8", errors="ignore")
    return raw

def read_vcf(file_like):
    text = _read_any(file_like)
    samples, records = None, []
    for line in text.splitlines():
        if line.startswith("##"): continue
        if line.startswith("#CHROM"):
            samples = line.rstrip("\n").split("\t")[9:]; continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 10: continue
        records.append((parts[0], int(parts[1]), parts[2], parts[3], parts[4], parts[9:]))
    if not records or samples is None: raise ValueError("VCF vide ou invalide")
    n_samp, n_snp = len(samples), len(records)
    gt = np.full((n_samp, n_snp), np.nan, dtype=np.float32)
    snp_rows = []
    for j, (ch, pos, sid, ref, alt, gts) in enumerate(records):
        for i, g in enumerate(gts):
            af = g.split(":")[0]
            sep = "/" if "/" in af else "|"
            al = af.split(sep)
            if len(al) < 2 or "." in al: continue
            try: gt[i, j] = int(al[0]) + int(al[1])
            except ValueError: pass
        snp_rows.append({"CHR": ch, "BP": pos, "SNP": sid, "A1": ref, "A2": alt, "CM": 0.0})
    return gt, pd.DataFrame(snp_rows), \
           pd.DataFrame({"FID": samples, "IID": samples, "POP": "POP"})

def read_plink(bed, bim, fam):
    fam_t = _read_any(fam); bim_t = _read_any(bim)
    ind_df = pd.DataFrame([l.split() for l in fam_t.strip().splitlines() if l.strip()],
                          columns=["FID","IID","PAT","MAT","SEX","PHENO"])[["FID","IID"]]
    ind_df["POP"] = "POP"
    snp_df = pd.DataFrame([l.split() for l in bim_t.strip().splitlines() if l.strip()],
                          columns=["CHR","SNP","CM","BP","A1","A2"])
    snp_df["BP"] = snp_df["BP"].astype(int)
    snp_df["CM"] = pd.to_numeric(snp_df["CM"], errors="coerce").fillna(0.0)
    raw = bed.read() if hasattr(bed, "read") else open(bed, "rb").read()
    if raw[:2] != b"\x6c\x1b":
        try: raw = gzip.decompress(raw)
        except Exception: pass
    if raw[:2] != b"\x6c\x1b": raise ValueError(".bed invalide")
    raw = raw[3:]
    n_ind, n_snp = len(ind_df), len(snp_df)
    nb = (n_ind + 3) // 4
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_snp, nb)
    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    for j in range(n_snp):
        for k in range(n_ind):
            c = (arr[j, k//4] >> (2*(k%4))) & 0b11
            gt[k, j] = {0b00: 2, 0b01: np.nan, 0b10: 1, 0b11: 0}[c]
    return gt, snp_df, ind_df

def read_ped(ped, mp):
    map_t = _read_any(mp)
    snp_df = pd.DataFrame([(l.split()[0], l.split()[1], 0.0,
                            int(l.split()[3]) if len(l.split()) > 3 else 0, "A", "G")
                           for l in map_t.strip().splitlines() if l.strip()],
                          columns=["CHR","SNP","CM","BP","A1","A2"])
    ped_t = _read_any(ped)
    lines = [l.split() for l in ped_t.strip().splitlines() if l.strip()]
    n_ind, n_snp = len(lines), len(snp_df)
    gt = np.full((n_ind, n_snp), np.nan, dtype=np.float32)
    ind_rows = []
    ref = snp_df.iloc[:, 4].tolist()
    for i, l in enumerate(lines):
        ind_rows.append({"FID": l[0], "IID": l[1]})
        al = l[6:]
        for j in range(n_snp):
            a1, a2 = al[2*j], al[2*j+1]
            if a1 == "0" or a2 == "0": continue
            gt[i, j] = 1 if a1 != a2 else (0 if a1 == ref[j] else 2)
    ind_df = pd.DataFrame(ind_rows); ind_df["POP"] = "POP"
    return gt, snp_df, ind_df

def read_csv_geno(file_like):
    name = getattr(file_like, "name", "")
    sep = "," if name.endswith(".csv") else "\t"
    df = pd.read_csv(file_like, sep=sep)
    id_like = {"SNP","ID","CHR","BP","A1","A2","CM"}
    snp_cols = [c for c in df.columns if c in id_like]
    if snp_cols:
        snp_df = df[snp_cols].reset_index(drop=True)
        sample_cols = [c for c in df.columns if c not in id_like]
        gt = df[sample_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        ind_df = pd.DataFrame({"FID": sample_cols, "IID": sample_cols, "POP": "POP"})
        for c, v in [("CHR","1"),("BP",None),("CM",0.0),("A1","A"),("A2","G"),("SNP",None)]:
            if c not in snp_df.columns:
                if c == "BP": snp_df[c] = np.arange(len(snp_df))*1000
                elif c == "SNP": snp_df[c] = [f"snp{i}" for i in range(len(snp_df))]
                else: snp_df[c] = v
        return gt, snp_df, ind_df
    ids = df.iloc[:, 0].astype(str).tolist()
    mat = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    snp_df = pd.DataFrame({"SNP": [f"snp{i}" for i in range(mat.shape[1])],
                           "CHR": "1", "BP": np.arange(mat.shape[1])*1000,
                           "CM": 0.0, "A1": "A", "A2": "G"})
    ind_df = pd.DataFrame({"FID": ids, "IID": ids, "POP": "POP"})
    return mat, snp_df, ind_df

def read_gff(file_like):
    text = _read_any(file_like)
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 9 or p[2] != "gene": continue
        gid = gname = "N/A"
        for kv in p[8].split(";"):
            kv = kv.strip()
            if kv.startswith("ID="): gid = kv[3:]
            elif kv.startswith("Name="): gname = kv[5:]
            elif kv.startswith("gene_id="): gid = kv[8:]
            elif kv.startswith("gene_name="): gname = kv[10:]
        rows.append({"CHR": p[0], "START": int(p[3]), "END": int(p[4]),
                     "GENE": gname if gname != "N/A" else gid,
                     "ID": gid, "STRAND": p[6]})
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# QC
# ------------------------------------------------------------
def qc_pipeline(gt, snp_df, ind_df, msi=0.2, mss=0.1, maf_min=0.05,
                hwe_p=1e-6, het_min=0.0, het_max=1.0):
    gt = gt.copy()
    miss_ind = np.isnan(gt).mean(1)
    miss_snp = np.isnan(gt).mean(0)
    het_vals = np.where(np.isnan(gt), np.nan, (gt == 1).astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        het_rate = np.nanmean(het_vals, axis=1)
    n_obs = (~np.isnan(gt)).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.where(n_obs > 0, np.nansum(gt, axis=0)/(2*n_obs), np.nan)
        maf = np.minimum(af, 1-af)
    hwe_pval = np.ones(gt.shape[1])
    for j in range(gt.shape[1]):
        col = gt[:, j]; col = col[~np.isnan(col)]
        if col.size < 10: continue
        n0 = (col==0).sum(); n1 = (col==1).sum(); n2 = (col==2).sum()
        n = n0+n1+n2; p = (2*n0+n1)/(2*n)
        if p in (0,1): continue
        q = 1-p
        exp0, exp1, exp2 = n*p*p, n*2*p*q, n*q*q
        if exp1 == 0 or exp0 == 0 or exp2 == 0: continue
        try:
            chi2 = (n0-exp0)**2/exp0 + (n1-exp1)**2/exp1 + (n2-exp2)**2/exp2
            hwe_pval[j] = stats.chi2.sf(chi2, 1)
        except Exception: pass
    keep_ind = (miss_ind <= msi) & (het_rate >= het_min) & (het_rate <= het_max)
    keep_ind = np.where(np.isnan(keep_ind), False, keep_ind)
    keep_snp = (miss_snp <= mss) & (maf >= maf_min) & (hwe_pval >= hwe_p)
    gt_f = gt[keep_ind][:, keep_snp]
    snp_f = snp_df.iloc[keep_snp].reset_index(drop=True)
    ind_f = ind_df.iloc[keep_ind].reset_index(drop=True)
    qstats = {
        "n_ind_keep": int(keep_ind.sum()), "n_snp_keep": int(keep_snp.sum()),
        "maf_mean": float(np.nanmean(maf[keep_snp])) if keep_snp.any() else 0.0,
        "call_rate": float(1 - np.nanmean(np.isnan(gt_f))) if gt_f.size else 0.0,
        "missing_ind": miss_ind.tolist(), "missing_snp": miss_snp.tolist(),
        "maf_values": maf[keep_snp].tolist() if keep_snp.any() else [],
    }
    return gt_f, snp_f, ind_f, qstats

# ============================================================
# GENOSCOPE - PARTIE 2/3 : Analyses
# ============================================================

def _impute_mean(gt):
    X = gt.copy()
    cm = np.nanmean(X, axis=0); cm = np.where(np.isnan(cm), 0, cm)
    inds = np.where(np.isnan(X)); X[inds] = np.take(cm, inds[1])
    return X

# ---------- PCA / MDS ----------
def pca_calc(gt, ind_df, n_pc=10):
    X = _impute_mean(gt)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Z = np.clip((X - mu)/sd, -3, 3)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    n_pc = min(n_pc, len(S))
    scores = U[:, :n_pc] * S[:n_pc]
    var = (S[:n_pc]**2)/(S**2).sum() * 100
    return scores, var, ind_df

def mds_calc(gt, ind_df):
    X = _impute_mean(gt)
    step = max(1, X.shape[1]//5000)
    D = squareform(pdist(X[:, ::step]))
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0)), ind_df

# ---------- Admixture (NMF) ----------
def _admixture_nmf(X, K, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    n, m = X.shape
    W = rng.random((n, K)) + 1e-3
    H = rng.random((K, m)) + 1e-3
    for _ in range(n_iter):
        W *= (X @ H.T) / (W @ H @ H.T + 1e-9)
        H *= (W.T @ X) / (W.T @ W @ H + 1e-9)
    return W / W.sum(1, keepdims=True)

def admix_pipeline(gt, ind_df, K_range, n_iter=100):
    X = np.clip(_impute_mean(gt), 0, 2)
    cv, Qs = [], {}
    for K in K_range:
        Q = _admixture_nmf(X, K, n_iter=n_iter)
        cv.append({"K": K, "CV": float(np.mean(np.var(Q, axis=0)))})
        Qs[K] = Q
    cv_df = pd.DataFrame(cv)
    best = int(cv_df.sort_values("CV", ascending=False)["K"].iloc[0])
    return cv_df, Qs[best], best, Qs

# ---------- LD ----------
def ld_decay(gt, snp_df, max_kb=500):
    X = _impute_mean(gt)
    chrom0 = str(snp_df["CHR"].iloc[0])
    idx = np.where(snp_df["CHR"].astype(str) == chrom0)[0][:500]
    if len(idx) < 2: idx = np.arange(min(200, X.shape[1]))
    snp_s = snp_df.iloc[idx].reset_index(drop=True)
    mu = X[:, idx].mean(0); sd = X[:, idx].std(0); sd[sd == 0] = 1
    Z = (X[:, idx] - mu)/sd
    R = np.corrcoef(Z, rowvar=False)
    rows = []
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            d = abs(snp_s.iloc[j]["BP"] - snp_s.iloc[i]["BP"])/1000.0
            if d > max_kb: continue
            rows.append({"snp1": snp_s.iloc[i]["SNP"], "snp2": snp_s.iloc[j]["SNP"],
                         "dist_kb": d, "r2": float(R[i, j]**2)})
    return pd.DataFrame(rows)

# ---------- ROH ----------
def roh_calc(gt, snp_df, ind_df, min_snps=20, max_gap_kb=1000):
    X = _impute_mean(gt)
    order = np.argsort([_chk(c) for c in snp_df["CHR"]])
    X = X[:, order]; snp_s = snp_df.iloc[order].reset_index(drop=True)
    roh_rows = []
    for i in range(X.shape[0]):
        start, run_chr = None, None
        for j in range(X.shape[1]):
            hom = X[i, j] in (0, 2)
            cj = str(snp_s.iloc[j]["CHR"])
            if hom:
                if start is None: start, run_chr = j, cj
                elif cj != run_chr:
                    if j - start >= min_snps: roh_rows.append((i, start, j-1, run_chr))
                    start, run_chr = j, cj
            else:
                if start is not None and j - start >= min_snps:
                    roh_rows.append((i, start, j-1, run_chr))
                start = None
        if start is not None and X.shape[1] - start >= min_snps:
            roh_rows.append((i, start, X.shape[1]-1, run_chr))
    rows = []
    for i, s, e, c in roh_rows:
        bp_s, bp_e = int(snp_s.iloc[s]["BP"]), int(snp_s.iloc[e]["BP"])
        if (bp_e - bp_s)/1000.0 > max_gap_kb: continue
        rows.append({"IID": ind_df.iloc[i]["IID"], "CHR": c,
                     "START": bp_s, "END": bp_e, "N_SNP": int(e-s+1),
                     "LENGTH_KB": round((bp_e-bp_s)/1000.0, 1)})
    roh_df = pd.DataFrame(rows)
    if roh_df.empty:
        return roh_df, pd.DataFrame({"IID": ind_df["IID"], "FROH": 0.0})
    genome_kb = max(1, (snp_df["BP"].max() - snp_df["BP"].min())/1000.0)
    fr = roh_df.groupby("IID")["LENGTH_KB"].sum().reset_index()
    fr["FROH"] = fr["LENGTH_KB"]/genome_kb
    return roh_df, fr

# ---------- Ne ----------
def ne_calc(ld_df, gt, snp_df):
    if ld_df is None or ld_df.empty:
        n = gt.shape[0]
        return pd.DataFrame({"generations": [1, 5, 10, 50, 100], "Ne": [n*2]*5})
    ld = ld_df.copy()
    bins = [0, 10, 25, 50, 100, 250, 500, 1000]
    ld["bin"] = pd.cut(ld["dist_kb"], bins=bins)
    m = ld.groupby("bin")["r2"].mean().reset_index()
    m["dist_kb"] = m["bin"].apply(lambda x: x.mid if hasattr(x, "mid") else np.nan)
    m = m.dropna(subset=["dist_kb"])
    if m.empty:
        return pd.DataFrame({"generations": [1, 5, 10, 50, 100], "Ne": [gt.shape[0]*2]*5})
    m["c"] = m["dist_kb"]/100000.0
    m["Ne"] = (1/(4*m["c"].replace(0, np.nan))) * (1/m["r2"].replace(0, np.nan) - 1)
    m["generations"] = 1/(2*m["c"].replace(0, np.nan))
    return m[["generations", "Ne"]].dropna().sort_values("generations").reset_index(drop=True)

# ---------- Reynolds ----------
def reynolds_calc(gt, ind_df, pop_col=None):
    pop_col = pop_col or ("POP" if "POP" in ind_df.columns else ind_df.columns[0])
    pops = ind_df[pop_col].astype(str).values
    names = sorted(set(pops))
    X = _impute_mean(gt)
    freqs = {p: X[pops == p].sum(0)/(2*(pops == p).sum()) for p in names}
    n = len(names); D = np.zeros((n, n))
    for i, pi in enumerate(names):
        for j, pj in enumerate(names):
            if i == j: continue
            p1, p2 = freqs[pi], freqs[pj]
            num, den = (p1-p2)**2, p1*(1-p2) + p2*(1-p1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.nanmean(np.where(den > 0, num/den, 0))
            D[i, j] = -np.log(1-r) if r < 1 else 10.0
    D = (D + D.T)/2
    if n >= 2:
        Z = linkage(squareform(D, checks=False), method="average")
        supp = np.ones(Z.shape[0]) * 100.0
    else:
        Z = np.zeros((1, 4)); supp = np.array([100.0])
    return D, names, Z, supp

# ---------- GWAS ----------
def gwas_pairwise(gt, snp_df, labels, ra, rb):
    mask = np.isin(labels, [ra, rb])
    if mask.sum() < 6: return pd.DataFrame()
    X = gt[mask]; y = (labels[mask] == rb).astype(float)
    rows = []
    for j in range(X.shape[1]):
        g = X[:, j]; valid = ~np.isnan(g)
        if valid.sum() < 6 or np.nanstd(g[valid]) == 0: continue
        gv, yv = g[valid], y[valid]
        n1, n0 = (yv == 1).sum(), (yv == 0).sum()
        if n1 < 2 or n0 < 2: continue
        p1, p0 = gv[yv == 1].mean()/2, gv[yv == 0].mean()/2
        p = gv.sum()/(2*len(gv))
        if p in (0, 1): continue
        se = np.sqrt(p*(1-p)*(1/n1 + 1/n0)*2)
        if se == 0: continue
        z = (p1 - p0)/se
        rows.append({"SNP": snp_df.iloc[j]["SNP"], "CHR": snp_df.iloc[j]["CHR"],
                     "BP": int(snp_df.iloc[j]["BP"]),
                     "P": float(2*stats.norm.sf(abs(z))),
                     "BETA": float(p1-p0), "Z": float(z)})
    return pd.DataFrame(rows).sort_values("P").reset_index(drop=True) if rows else pd.DataFrame()

@cache_data
def permutation_test_race(gt, snp_df, labels, ra, rb, n_perm=100, seed=42, max_snp=5000):
    rng = np.random.default_rng(seed)
    n_avail = min(gt.shape[1], snp_df.shape[0])
    if n_avail > max_snp:
        idx = np.sort(rng.choice(n_avail, max_snp, replace=False))
        snp_df = snp_df.iloc[idx].reset_index(drop=True); gt = gt[:, idx]
    original = gwas_pairwise(gt, snp_df, labels, ra, rb)
    if original.empty: return pd.DataFrame(), 1.0, 0, []
    n_obs = int((original["P"] < 0.05/max(len(original), 1)).sum())
    mask = np.isin(labels, [ra, rb])
    labels_sub, gt_sub = labels[mask].copy(), gt[mask]
    perm_counts = []
    for _ in range(n_perm):
        rng.shuffle(labels_sub)
        r = gwas_pairwise(gt_sub, snp_df, labels_sub, ra, rb)
        perm_counts.append(int((r["P"] < 0.05/max(len(r), 1)).sum()) if not r.empty else 0)
    p_emp = (np.sum(np.array(perm_counts) >= n_obs) + 1)/(n_perm + 1)
    return original, float(p_emp), n_obs, perm_counts

# ---------- GBLUP (CORRIGÉ : h2 + n_pc) ----------
def gblup_calc(gt, ind_df, pheno_df, h2=0.5, n_pc=0):
    """GBLUP avec blending h² et covariables PC optionnelles."""
    X = _impute_mean(gt)
    p = np.clip(X.mean(0) / 2, 0.01, 0.99)
    denom = 2 * np.sum(p * (1 - p)) or 1.0
    G = ((X - 2 * p) @ (X - 2 * p).T) / denom

    id_col = "IID" if "IID" in pheno_df.columns else pheno_df.columns[0]
    y_col = "Phenotype" if "Phenotype" in pheno_df.columns else pheno_df.columns[1]
    ph = pheno_df[[id_col, y_col]].rename(
        columns={id_col: "IID", y_col: "Phenotype"})
    m = ind_df[["IID"]].merge(ph, on="IID", how="inner")
    if m.empty:
        raise ValueError("Aucun IID commun entre ind_df et pheno_df")

    idx = [ind_df.index[ind_df["IID"] == i][0] for i in m["IID"]]
    G_sub = G[np.ix_(idx, idx)]
    y = m["Phenotype"].values.astype(float)

    # Covariables fixes : intercept + PCs optionnels
    X_fixed = np.ones((len(y), 1))
    if n_pc > 0:
        X_pca = _impute_mean(gt[idx])
        X_pca = X_pca - X_pca.mean(0)
        U, S, _ = np.linalg.svd(X_pca, full_matrices=False)
        pcs = U[:, :n_pc] * S[:n_pc]
        X_fixed = np.column_stack([X_fixed, pcs])

    # Henderson : lambda = (1-h²)/h²
    lam = (1 - h2) / h2 if h2 > 0 else 1.0
    try:
        G_inv = np.linalg.inv(G_sub + lam * np.eye(len(y)))
        beta = np.linalg.lstsq(X_fixed, y, rcond=None)[0]
        resid = y - X_fixed @ beta
        gebv = G_inv @ resid
    except np.linalg.LinAlgError:
        gebv = np.zeros(len(y))

    return pd.DataFrame({"IID": m["IID"], "Phenotype": y, "GEBV": gebv})

# ---------- Diversité ----------
def diversity_per_snp(gt, snp_df):
    n_obs = (~np.isnan(gt)).sum(0).astype(float)
    cnt1 = (gt == 1).sum(0) + 2*(gt == 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(n_obs > 0, cnt1/(2*n_obs), np.nan)
        p, q = np.maximum(1-f1, f1), np.minimum(1-f1, f1)
        he = 2*p*q
        ho = np.where(n_obs > 0, (gt == 1).sum(0)/n_obs, np.nan)
        pic = 1 - (p**2 + q**2) - 2*(p**2)*(q**2)
        fis = 1 - ho/he
    df = pd.DataFrame({"N": n_obs.astype(int), "MAF_div": q, "Ho": ho,
                       "He": he, "Nei_D": he, "PIC": pic, "Fis": fis})
    for c in ["SNP", "CHR", "BP"]:
        if c in snp_df.columns: df[c] = snp_df[c].values[:len(df)]
    return df

def diversity_per_pop(gt, ind_df, pop_col=None):
    pop_col = pop_col or ("POP" if "POP" in ind_df.columns else ind_df.columns[0])
    pops = ind_df[pop_col].astype(str).values
    out = []
    for pop in sorted(set(pops)):
        m = pops == pop
        d = diversity_per_snp(gt[m], pd.DataFrame({"SNP": range(gt.shape[1])}))
        if d.empty: continue
        out.append({"Population": pop, "N_ind": int(m.sum()),
                    "Ho_mean": d["Ho"].mean(), "He_mean": d["He"].mean(),
                    "Nei_D_mean": d["Nei_D"].mean(), "PIC_mean": d["PIC"].mean(),
                    "Fis_mean": d["Fis"].mean(), "MAF_mean": d["MAF_div"].mean()})
    return pd.DataFrame(out)

def enrich_and_filter_diversity(gt, snp_df, pic_min=0.0, drop_monomorphic=True):
    div = diversity_per_snp(gt, snp_df)
    if div.empty:
        return gt, snp_df, {"n_snp_before": gt.shape[1], "n_snp_after": gt.shape[1],
                            "n_dropped": 0, "Ho_mean": 0, "He_mean": 0,
                            "PIC_mean": 0, "PIC_high": 0, "PIC_info": 0}
    snp_out = snp_df.copy().reset_index(drop=True)
    for col in ["Ho", "He", "Nei_D", "PIC", "Fis", "MAF_div", "N"]:
        snp_out[col] = div[col].values
    mask = np.ones(len(snp_out), dtype=bool)
    if drop_monomorphic: mask &= snp_out["MAF_div"].fillna(0).values > 0
    if pic_min > 0: mask &= snp_out["PIC"].fillna(0).values >= pic_min
    snp_k, gt_k = snp_out[mask].reset_index(drop=True), gt[:, mask]
    st_d = {"n_snp_before": int(gt.shape[1]), "n_snp_after": int(gt_k.shape[1]),
            "n_dropped": int((~mask).sum()),
            "Ho_mean": float(snp_k["Ho"].mean()) if len(snp_k) else 0,
            "He_mean": float(snp_k["He"].mean()) if len(snp_k) else 0,
            "PIC_mean": float(snp_k["PIC"].mean()) if len(snp_k) else 0,
            "PIC_high": int((snp_k["PIC"] >= 0.5).sum()),
            "PIC_info": int((snp_k["PIC"] >= 0.25).sum())}
    return gt_k, snp_k, st_d

# ---------- Méta-analyse ----------
def _fisher_combine(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0)
    return float(stats.chi2.sf(-2*np.sum(np.log(p)), df=2*len(p)))

def combine_pvalues_fisher(r1, r2):
    if r1 is None or r2 is None or r1.empty or r2.empty: return None
    m = r1[["SNP", "CHR", "BP", "P"]].rename(columns={"P": "P_race"}).merge(
        r2[["SNP", "P"]].rename(columns={"P": "P_group"}), on="SNP")
    if m.empty: return None
    m["P_combined"] = [_fisher_combine(x) for x in m[["P_race", "P_group"]].values]
    return m.sort_values("P_combined").reset_index(drop=True)

# ---------- VEP / Annotation ----------
_SEVERITY = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}

def traduire_consequence_vep(c):
    c = str(c).lower()
    if "missense" in c: return "🔴 Exon (Faux-sens)"
    if "synonymous" in c: return "🟡 Exon (Silencieuse)"
    if "stop_gained" in c or "frameshift" in c: return "⛔ Exon (Non-sens)"
    if "splice" in c: return "⚠️ Jonction Exon/Intron"
    if "intron" in c: return "🟢 Intron"
    if "3_prime_utr" in c: return "🔵 3'UTR"
    if "5_prime_utr" in c: return "🔵 5'UTR"
    if "upstream" in c: return "🟣 Promoteur amont"
    if "downstream" in c: return "🟣 Régulateur aval"
    return "⚪ Intergénique"

def run_vep_api(variants, species, batch=200):
    if not variants: return pd.DataFrame()
    url = f"https://rest.ensembl.org/vep/{species}/region?regulatory=1"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    results = []
    for i in range(0, len(variants), batch):
        try:
            r = requests.post(url, headers=headers,
                              data=json.dumps({"variants": variants[i:i+batch]}), timeout=60)
            if r.status_code != 200: continue
            for item in r.json():
                sid = item.get("input", "N/A")
                if item.get("transcript_consequences"):
                    cons = sorted(item["transcript_consequences"],
                                  key=lambda x: _SEVERITY.get(x.get("impact", "MODIFIER"), 0),
                                  reverse=True)[0]
                    results.append({"SNP": sid, "GENE": cons.get("gene_symbol", "N/A"),
                                    "CONSEQUENCE": ", ".join(cons.get("consequence_terms", ["N/A"])),
                                    "IMPACT": cons.get("impact", "N/A"),
                                    "AMINO_ACID_CHANGE": cons.get("amino_acids", "N/A"),
                                    "SIFT": cons.get("sift_prediction", "N/A"),
                                    "POLYPHEN": cons.get("polyphen_prediction", "N/A")})
                else:
                    results.append({"SNP": sid, "GENE": "Intergénique",
                                    "CONSEQUENCE": "N/A", "IMPACT": "MODIFIER",
                                    "AMINO_ACID_CHANGE": "N/A", "SIFT": "N/A", "POLYPHEN": "N/A"})
        except Exception: pass
        time.sleep(0.5)
    return pd.DataFrame(results)

def annotate_genes_cm(snp_target, snp_full, gff, window_kb=100):
    w = window_kb * 1000
    rows = []
    for _, s in snp_target.iterrows():
        ch, pos = str(s["CHR"]), int(s.get("BP", 0))
        gs = gff[(gff["CHR"].astype(str) == ch) &
                 (gff["END"] >= pos-w) & (gff["START"] <= pos+w)]
        for _, g in gs.iterrows():
            dist = min(abs(g["START"]-pos), abs(g["END"]-pos))/1000.0
            rows.append({"SNP": s.get("SNP"), "CHR": ch, "SNP_POS_BP": pos,
                         "GENE": g["GENE"], "GENE_START": int(g["START"]),
                         "GENE_END": int(g["END"]), "DIST_KB": round(dist, 2),
                         "WINDOW": f"±{window_kb} kb"})
    return pd.DataFrame(rows)

# ---------- vep_annotate : WRAPPER UNIFIÉ (NOUVEAU) ----------
def vep_annotate(snp_sub, species, use_api=False, gff=None):
    """Wrapper unifié : VEP API ou annotation GFF locale."""
    if snp_sub is None or snp_sub.empty:
        return pd.DataFrame()

    if use_api:
        variants = [
            f"{r['CHR']} {int(r['BP'])} . {r.get('A1', 'A')} {r.get('A2', 'G')} . . ."
            for _, r in snp_sub.iterrows()
        ]
        df = run_vep_api(variants, species)
        if df.empty:
            return pd.DataFrame({"SNP": snp_sub["SNP"].tolist(),
                                 "Gene": "N/A", "Consequence": "N/A",
                                 "Impact": "N/A", "CHR": snp_sub["CHR"].tolist()})
        df = df.rename(columns={"GENE": "Gene",
                                "CONSEQUENCE": "Consequence",
                                "IMPACT": "Impact",
                                "AMINO_ACID_CHANGE": "AA_change"})
        df["SNP"] = snp_sub["SNP"].values[:len(df)]
        df["CHR"] = snp_sub["CHR"].values[:len(df)]
        return df

    if gff is not None and not gff.empty:
        ann = annotate_genes_cm(snp_sub, snp_sub, gff, window_kb=100)
        if ann.empty:
            return pd.DataFrame({"SNP": snp_sub["SNP"].tolist(),
                                 "Gene": "N/A", "Consequence": "Intergénique",
                                 "Impact": "MODIFIER", "CHR": snp_sub["CHR"].tolist()})
        ann = ann.rename(columns={"GENE": "Gene", "SNP_POS_BP": "BP"})
        ann["Consequence"] = "Région génique (" + ann["WINDOW"] + ")"
        ann["Impact"] = "MODIFIER"
        return ann[["SNP", "CHR", "BP", "Gene", "Consequence", "Impact",
                    "GENE_START", "GENE_END", "DIST_KB"]]

    return pd.DataFrame({"SNP": snp_sub["SNP"].tolist(),
                         "Gene": "N/A", "Consequence": "N/A",
                         "Impact": "N/A", "CHR": snp_sub["CHR"].tolist()})

# ---------- Enrichissement ----------
def run_enrichment(gene_list, gene_sets=None, organism="Human"):
    try:
        import gseapy as gp
        genes = list({str(g).strip() for g in gene_list
                      if str(g).strip() and str(g) != "nan"})
        if len(genes) < 3: return pd.DataFrame()
        gene_sets = gene_sets or ["GO_Biological_Process_2023", "KEGG_2021_Human"]
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets,
                         organism=organism, cutoff=1.0, outdir=None)
        df = enr.results.copy()
        for old, new in {"Term": "TERM", "Adjusted P-value": "FDR",
                          "P-value": "PVAL", "Overlap": "OVERLAP",
                          "Genes": "GENES", "Combined Score": "SCORE"}.items():
            if old in df.columns: df = df.rename(columns={old: new})
        return df.sort_values("FDR").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# ---------- enrich_analysis : WRAPPER UNIFIÉ (NOUVEAU) ----------
def enrich_analysis(gene_list, database="GO_BP", p_threshold=0.05):
    """Wrapper normalisé pour run_enrichment (Enrichr)."""
    db_map = {
        "GO_BP":    ["GO_Biological_Process_2023"],
        "GO_MF":    ["GO_Molecular_Function_2023"],
        "GO_CC":    ["GO_Cellular_Component_2023"],
        "KEGG":     ["KEGG_2021_Human"],
        "Reactome": ["Reactome_2022"],
    }
    gene_sets = db_map.get(database, ["GO_Biological_Process_2023"])
    df = run_enrichment(gene_list, gene_sets=gene_sets, organism="Human")
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "TERM": "Term", "FDR": "fdr", "PVAL": "p_value",
        "OVERLAP": "overlap", "GENES": "genes", "SCORE": "score",
    })
    if "p_value" in df.columns and p_threshold < 1.0:
        df = df[df["p_value"] <= p_threshold].reset_index(drop=True)
    return df

def query_string_network(genes, species, required_score=400):
    if len(genes) < 2: return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([str(g).strip() for g in genes if str(g).strip()])
    url = (f"https://string-db.org/api/json/network?identifiers={ids}"
           f"&species={species}&required_score={required_score}")
    try:
        r = requests.get(url, timeout=60); r.raise_for_status()
        data = r.json()
        if not data: return pd.DataFrame(), pd.DataFrame()
        df = pd.DataFrame(data)
        edges = df[["preferredName_A", "preferredName_B", "score"]].copy()
        edges.columns = ["SOURCE", "TARGET", "SCORE"]
        nodes = pd.DataFrame({"ID": sorted(set(edges["SOURCE"]) | set(edges["TARGET"]))})
        return edges, nodes
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ---------- Export ----------
def build_plink(gt, ind_df, snp_df, prefix="genome"):
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        n, m = gt.shape; nb = (n+3)//4
        buf = bytearray([0x6C, 0x1B, 0x01])
        for j in range(m):
            bits = np.zeros(n, dtype=np.uint8)
            bits[np.isnan(gt[:, j])] = 0b01
            bits[gt[:, j] == 0] = 0b11
            bits[gt[:, j] == 1] = 0b10
            bits[gt[:, j] == 2] = 0b00
            pk = np.zeros(nb, dtype=np.uint8)
            for k in range(n): pk[k//4] |= (bits[k] & 0x03) << (2*(k % 4))
            buf.extend(pk.tobytes())
        zf.writestr(f"{prefix}.bed", bytes(buf))
        zf.writestr(f"{prefix}.bim", "\n".join(
            f"{r['CHR']}\t{r['SNP']}\t{r.get('CM',0)}\t{int(r['BP'])}\tA\tG"
            for _, r in snp_df.iterrows()) + "\n")
        zf.writestr(f"{prefix}.fam", "\n".join(
            f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9" for _, r in ind_df.iterrows()) + "\n")
    return bio.getvalue()

def build_vcf(gt, ind_df, snp_df, proj="Genome"):
    h = ["##fileformat=VCFv4.2", f"##source={proj}",
         '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
         "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
         + "\t".join(ind_df["IID"].astype(str))]
    b = []
    for j in range(gt.shape[1]):
        r = snp_df.iloc[j]
        gts = ["./." if np.isnan(gt[i, j]) else
               ("0/0" if gt[i, j] == 0 else "0/1" if gt[i, j] == 1 else "1/1")
               for i in range(gt.shape[0])]
        ref = r.get("A1", "A") if r.get("A1", "A") not in (".", "0", "") else "A"
        alt = r.get("A2", "G") if r.get("A2", "G") not in (".", "0", "") else "G"
        b.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref}\t{alt}\t.\tPASS\t.\tGT\t"
                 + "\t".join(gts))
    return "\n".join(h + b) + "\n"

# ============================================================
# GENOSCOPE - PARTIE 3/3 : Plots + Interface
# ============================================================

# ---------- Plots ----------
def plot_missing(mi, ms):
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind", "Missing/SNP"))
    f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1)
    f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False)
    return f

def plot_pca(s, v, l):
    df = pd.DataFrame({"PC1": s[:, 0], "PC2": s[:, 1], "Pop": list(l)})
    return px.scatter(df, x="PC1", y="PC2", color="Pop",
                      title=f"PCA — PC1({v[0]:.1f}%) / PC2({v[1]:.1f}%)", height=550)

def plot_mds(m, l):
    df = pd.DataFrame({"MDS1": m[:, 0], "MDS2": m[:, 1], "Pop": list(l)})
    return px.scatter(df, x="MDS1", y="MDS2", color="Pop",
                      title="MDS (Reynolds)", height=520)

def plot_ld(ld, b=20):
    if ld is None or ld.empty: return None
    d = ld.copy()
    d["bin"] = (d["dist_kb"]//b)*b
    return px.line(d.groupby("bin")["r2"].mean().reset_index(),
                   x="bin", y="r2", title="LD decay", height=450)

def plot_admix(Q, labs, pops, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
    df["IID"] = labs; df["Pop"] = pops; df["_d"] = Q.argmax(1)
    df["_po"] = pd.Categorical(pops, categories=sorted(set(pops)))
    df = df.sort_values(["_po", "_d"]).reset_index(drop=True); df["x"] = np.arange(len(df))
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k+1}"], name=f"K{k+1}",
                             marker_color=px.colors.qualitative.Set2[k % 8]))
    fig.update_layout(barmode="stack", title=f"Admixture K={K}", height=500)
    return fig, df

def plot_tree(Z, labels, support=None, title="", mid=None):
    n = len(labels); labels = [str(l) for l in labels]
    lo = leaves_list(Z); ly = {int(l): i for i, l in enumerate(lo)}
    cy, cx = {i: ly[i] for i in range(n)}, {i: 0.0 for i in range(n)}
    fig = go.Figure()
    for k, row in enumerate(Z):
        i, j, d = int(row[0]), int(row[1]), float(row[2])
        col = "#3498db"
        if support is not None and k < len(support):
            s = support[k]
            col = "#2ecc71" if s >= 70 else "#f39c12" if s >= 50 else "#e74c3c"
        fig.add_trace(go.Scatter(x=[cx[i], d], y=[cy[i], cy[i]], mode="lines",
                                 line=dict(color=col, width=2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[cx[j], d], y=[cy[j], cy[j]], mode="lines",
                                 line=dict(color=col, width=2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[d, d], y=[cy[i], cy[j]], mode="lines",
                                 line=dict(color=col, width=2), showlegend=False, hoverinfo="skip"))
        ni = n+k; cy[ni] = (cy[i]+cy[j])/2; cx[ni] = d
    for leaf, y in ly.items():
        fig.add_annotation(x=0, y=y, text=f"<b>{labels[leaf]}</b>",
                           showarrow=False, xanchor="right", xshift=-8,
                           font=dict(size=12, color="#2c3e50"))
    fig.update_layout(title=title, xaxis_title="Distance Reynolds",
                      yaxis=dict(showticklabels=False, showgrid=False),
                      height=max(500, n*35), plot_bgcolor="white")
    return fig

def plot_manhattan(df, yc="P", title="", thr=None):
    if df is None or df.empty: return None
    df = df.copy()
    if yc == "P":
        df["_y"] = -np.log10(np.clip(df["P"], 1e-300, 1)); yc = "_y"
    df = df[np.isfinite(df[yc])].copy()
    df = df.assign(_k=df["CHR"].map(lambda x: _chk(x)[1])).sort_values(["_k", "BP"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub.index, y=sub[yc], mode="markers",
                                 marker=dict(size=5,
                                             color="#2c3e50" if i % 2 == 0 else "#7f8c8d"),
                                 text=sub["SNP"], hoverinfo="text", showlegend=False))
    if thr: fig.add_hline(y=thr, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="Chr", yaxis_title=yc, height=500)
    return fig

def plot_gblup(df):
    fig = px.scatter(df, x="Phenotype", y="GEBV", title="GBLUP")
    fig.update_traces(marker=dict(size=10, color="royalblue"))
    if len(df) > 1:
        try:
            z = np.polyfit(df["Phenotype"], df["GEBV"], 1)
            p = np.poly1d(z)
            xr = np.linspace(df["Phenotype"].min(), df["Phenotype"].max(), 100)
            fig.add_trace(go.Scatter(x=xr, y=p(xr), mode="lines", name="OLS",
                                     line=dict(color="red", dash="dash")))
        except Exception: pass
    return fig

def plot_string_network(edges, nodes):
    if edges is None or edges.empty: return None
    try: import networkx as nx
    except ImportError: st.error("⚠️ pip install networkx"); return None
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r["SOURCE"], r["TARGET"], weight=float(r["SCORE"]))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    ex, ey = [], []
    for s, t in G.edges():
        x0, y0 = pos[s]; x1, y1 = pos[t]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(width=0.6, color="#cccccc"),
                             hoverinfo="skip", showlegend=False))
    nx_, ny_, deg = [], [], []
    for n in G.nodes():
        x, y = pos[n]; nx_.append(x); ny_.append(y); deg.append(G.degree(n))
    fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers+text",
                             marker=dict(size=[8+3*d for d in deg], color=deg,
                                         colorscale="Viridis",
                                         line=dict(width=1, color="white")),
                             text=list(G.nodes()), textposition="top center"))
    fig.update_layout(showlegend=False, height=600,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# ---------- Session state ----------
_KEYS = ["gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt",
         "qc_stats", "pca_s", "pca_v", "ind_pca", "mds_c", "ind_mds",
         "admix_Q", "admix_K", "admix_all", "cv_res", "ld_df", "roh_df",
         "froh", "ne_dict", "reyn_D", "reyn_pops", "tree_Z", "tree_support",
         "gwas_res", "gwas_pheno", "gblup_df", "meta_res", "vep_res",
         "snp_annot", "enrich_res", "pheno_df", "gff", "species", "taxon",
         "string_edges", "string_nodes", "div_stats", "div_pop"]

for _k in _KEYS:
    st.session_state.setdefault(_k, None)

def _g(k, d=None):
    v = st.session_state.get(k)
    return v if v is not None else d

def _s(k, v): st.session_state[k] = v

def _ok(v):
    if v is None: return False
    if isinstance(v, pd.DataFrame): return not v.empty
    if isinstance(v, (list, tuple, dict, np.ndarray, pd.Series)): return len(v) > 0
    return True

def _need(key, label="Données"):
    if not _ok(_g(key)):
        st.warning(f"⚠️ {label} manquantes."); return False
    return True

# ---------- Sidebar (FIX #4 : ternaire au lieu de `or` numpy) ----------
def render_sidebar():
    st.sidebar.markdown("# 🧬 GenoScope")
    pages = {
        "📥 Chargement": "load", "🧪 QC & Filtres": "qc",
        "🧮 Diversité": "div", "📊 PCA & MDS": "pca",
        "🧬 Admixture": "admix", "🔗 LD & ROH": "ld",
        "📉 Ne & Distances": "ne", "🌳 Arbre & Réseau": "tree",
        "🎯 GWAS": "gwas", "🔀 Méta-analyse": "meta",
        "🔬 VEP & Annotation": "vep", "🧩 Enrichissement": "enrich",
        "🏗️ GBLUP": "gblup", "📤 Export": "export",
    }
    choice = st.sidebar.radio("Navigation", list(pages.keys()))
    st.sidebar.divider()

    # FIX #4 : plus de `or` sur des ndarray → ValueError numpy ambiguous
    gt_display = _g("gt_filt") if _ok(_g("gt_filt")) else _g("gt")
    if _ok(gt_display):
        st.sidebar.markdown("### 📊 Jeu actif")
        st.sidebar.metric("Individus", gt_display.shape[0])
        st.sidebar.metric("SNPs", gt_display.shape[1])
        st.sidebar.metric("Missing", f"{100*np.isnan(gt_display).mean():.2f} %")
    return pages[choice]

# ---------- Pages ----------
def page_load():
    st.header("📥 Chargement des données")
    c1, c2 = st.columns([1, 2])
    with c1:
        fmt = st.selectbox("Format", ["VCF", "PLINK", "PED/MAP", "CSV/TSV"])
        sp = st.selectbox("Espèce",
                          ["human", "mouse", "cattle", "sheep", "goat",
                           "pig", "horse", "dog", "chicken", "zebrafish"])
        _s("species", sp)
        _s("taxon", {"human": 9606, "mouse": 10090, "cattle": 9913,
                     "sheep": 9940, "goat": 9925, "pig": 9823,
                     "horse": 9796, "dog": 9615, "chicken": 9031,
                     "zebrafish": 7955}.get(sp))
    with c2:
        try:
            if fmt == "VCF":
                f = st.file_uploader("VCF", type=["vcf", "gz"])
                if f and st.button("Charger", type="primary"):
                    gt, snp, ind = read_vcf(f)
                    for k, v in [("gt", gt), ("snp_df", snp), ("ind_df", ind)]: _s(k, v)
                    st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")
            elif fmt == "PLINK":
                b = st.file_uploader(".bed", type=["bed"])
                m = st.file_uploader(".bim", type=["bim"])
                fa = st.file_uploader(".fam", type=["fam"])
                if b and m and fa and st.button("Charger", type="primary"):
                    gt, snp, ind = read_plink(b, m, fa)
                    for k, v in [("gt", gt), ("snp_df", snp), ("ind_df", ind)]: _s(k, v)
                    st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")
            elif fmt == "PED/MAP":
                p = st.file_uploader(".ped", type=["ped"])
                mp = st.file_uploader(".map", type=["map"])
                if p and mp and st.button("Charger", type="primary"):
                    gt, snp, ind = read_ped(p, mp)
                    for k, v in [("gt", gt), ("snp_df", snp), ("ind_df", ind)]: _s(k, v)
                    st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")
            else:
                f = st.file_uploader("CSV/TSV", type=["csv", "tsv", "txt"])
                if f and st.button("Charger", type="primary"):
                    gt, snp, ind = read_csv_geno(f)
                    for k, v in [("gt", gt), ("snp_df", snp), ("ind_df", ind)]: _s(k, v)
                    st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")
        except Exception as e:
            st.error(f"❌ {e}")

    st.divider()
    g = st.file_uploader("GFF3 (optionnel)", type=["gff", "gff3", "gz"])
    if g and st.button("Charger GFF"):
        try: _s("gff", read_gff(g)); st.success("✅ GFF chargé")
        except Exception as e: st.error(f"❌ {e}")

    if _ok(_g("gt")):
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", _g("gt").shape[0])
        c2.metric("SNPs", _g("gt").shape[1])
        c3.metric("Missing", f"{100*np.isnan(_g('gt')).mean():.2f} %")
        t1, t2 = st.tabs(["SNPs", "Individus"])
        with t1: st.dataframe(_g("snp_df").head(20), use_container_width=True)
        with t2: st.dataframe(_g("ind_df").head(20), use_container_width=True)

def page_qc():
    st.header("🧪 QC & Filtres")
    if not _need("gt"): return
    c1, c2, c3 = st.columns(3)
    with c1:
        msi = st.slider("Missing/Ind max", 0.0, 1.0, 0.20, 0.01)
        maf_min = st.slider("MAF min", 0.0, 0.5, 0.05, 0.01)
    with c2:
        mss = st.slider("Missing/SNP max", 0.0, 1.0, 0.10, 0.01)
        hwe_p = st.number_input("HWE p min", 0.0, 1.0, 1e-6, format="%.1e")
    with c3:
        het_min = st.slider("Het min", 0.0, 1.0, 0.0, 0.01)
        het_max = st.slider("Het max", 0.0, 1.0, 1.0, 0.01)

    if st.button("▶️ Lancer QC", type="primary"):
        try:
            gt_f, snp_f, ind_f, stats_ = qc_pipeline(
                _g("gt"), _g("snp_df"), _g("ind_df"),
                msi=msi, mss=mss, maf_min=maf_min, hwe_p=hwe_p,
                het_min=het_min, het_max=het_max)
            _s("gt_filt", gt_f); _s("snp_filt", snp_f)
            _s("ind_filt", ind_f); _s("qc_stats", stats_)
            st.success(f"✅ {gt_f.shape[0]} ind × {gt_f.shape[1]} SNPs")
        except Exception as e: st.error(f"❌ {e}")

    if _ok(_g("qc_stats")):
        q = _g("qc_stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ind", q["n_ind_keep"])
        c2.metric("SNPs", q["n_snp_keep"])
        c3.metric("MAF moyen", f"{q['maf_mean']:.3f}")
        c4.metric("Call rate", f"{q['call_rate']*100:.1f}%")
        st.plotly_chart(plot_missing(q["missing_ind"], q["missing_snp"]),
                        use_container_width=True)

    if _ok(_g("gt_filt")):
        st.divider()
        st.subheader("🧮 Diversité & filtre informatif")
        c1, c2 = st.columns(2)
        with c1:
            pic_min = st.slider("PIC minimum", 0.0, 0.5, 0.0, 0.01,
                                help="0=off · 0.25=informatif · 0.5=très informatif")
        with c2:
            drop_mono = st.checkbox("Supprimer monomorphes", value=True)
        if st.button("▶️ Enrichir & filtrer (Ho/He/PIC/Fis)"):
            try:
                gt_e, snp_e, st_d = enrich_and_filter_diversity(
                    _g("gt_filt"), _g("snp_filt"),
                    pic_min=float(pic_min), drop_monomorphic=drop_mono)
                _s("gt_filt", gt_e); _s("snp_filt", snp_e); _s("div_stats", st_d)
                st.success(f"✅ {st_d['n_snp_after']} SNPs — PIC moy={st_d['PIC_mean']:.3f}")
            except Exception as e: st.error(f"❌ {e}")
        if _ok(_g("div_stats")):
            s = _g("div_stats")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ho moyen", f"{s['Ho_mean']:.3f}")
            c2.metric("He moyen", f"{s['He_mean']:.3f}")
            c3.metric("PIC moyen", f"{s['PIC_mean']:.3f}")
            c4.metric("Très informatifs (PIC≥0.5)", s["PIC_high"])

def page_diversity():
    st.header("🧮 Diversité génétique")
    if not _need("snp_filt"): return
    snp = _g("snp_filt")
    if not all(c in snp.columns for c in ["Ho", "He", "PIC"]):
        st.info("ℹ️ Lancer « Enrichir & filtrer » dans QC.")
        return
    st.subheader("📈 Indices / SNP")
    m = snp.melt(id_vars=["SNP"], value_vars=["Ho", "He", "PIC", "Fis", "MAF_div"],
                 var_name="Indice", value_name="Valeur")
    st.plotly_chart(px.histogram(m, x="Valeur", facet_col="Indice", nbins=50,
                                 color="Indice"), use_container_width=True)
    st.subheader("🎨 Classes PIC")
    cls = pd.cut(snp["PIC"], bins=[0, 0.25, 0.5, 1.0],
                 labels=["Faible", "Informatif", "Très informatif"])
    cc = cls.value_counts().reset_index()
    cc.columns = ["Classe", "N"]
    st.plotly_chart(px.pie(cc, names="Classe", values="N", hole=0.4,
                           title="Répartition PIC"), use_container_width=True)
    st.subheader("🌍 Moyennes par population")
    if st.button("▶️ Calculer / population"):
        try:
            _s("div_pop", diversity_per_pop(_g("gt_filt"), _g("ind_filt")))
            st.success("✅")
        except Exception as e: st.error(f"❌ {e}")
    if _ok(_g("div_pop")):
        dp = _g("div_pop")
        st.dataframe(dp.round(4), use_container_width=True)
        dm = dp.melt(id_vars="Population",
                     value_vars=["Ho_mean", "He_mean", "PIC_mean", "Fis_mean"],
                     var_name="Indice", value_name="Valeur")
        st.plotly_chart(px.bar(dm, x="Population", y="Valeur", color="Indice",
                               barmode="group",
                               color_discrete_sequence=px.colors.qualitative.Set2),
                        use_container_width=True)
    st.download_button("⬇️ CSV diversité / SNP",
                        snp[["SNP", "CHR", "BP", "Ho", "He", "Nei_D", "PIC", "Fis", "MAF_div"]]
                        .to_csv(index=False).encode(),
                        "diversite.csv", "text/csv")

def page_pca():
    st.header("📊 PCA & MDS")
    if not _need("gt_filt"): return
    c1, c2 = st.columns(2)
    with c1:
        n_pc = st.slider("Nb PC", 2, 20, 10)
        if st.button("▶️ PCA", type="primary"):
            try:
                s, v, ind = pca_calc(_g("gt_filt"), _g("ind_filt"), n_pc)
                _s("pca_s", s); _s("pca_v", v); _s("ind_pca", ind)
                st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
    with c2:
        if st.button("▶️ MDS", type="primary"):
            try:
                m, ind = mds_calc(_g("gt_filt"), _g("ind_filt"))
                _s("mds_c", m); _s("ind_mds", ind); st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
    if _ok(_g("pca_s")):
        ind = _g("ind_pca"); pc = "POP" if "POP" in ind.columns else ind.columns[0]
        st.plotly_chart(plot_pca(_g("pca_s"), _g("pca_v"), ind[pc].astype(str)),
                        use_container_width=True)
        st.plotly_chart(px.bar(x=[f"PC{i+1}" for i in range(len(_g("pca_v")))],
                               y=_g("pca_v"), title="Variance expliquée (%)",
                               color=_g("pca_v"), color_continuous_scale="Viridis"),
                        use_container_width=True)
    if _ok(_g("mds_c")):
        ind = _g("ind_mds"); pc = "POP" if "POP" in ind.columns else ind.columns[0]
        st.plotly_chart(plot_mds(_g("mds_c"), ind[pc].astype(str)),
                        use_container_width=True)

def page_admix():
    st.header("🧬 Admixture")
    if not _need("gt_filt"): return
    c1, c2 = st.columns(2)
    with c1:
        k_min = st.number_input("K min", 2, 20, 2)
        k_max = st.number_input("K max", 2, 20, 5)
    with c2:
        n_iter = st.number_input("Itérations", 10, 500, 100)
        if st.button("▶️ Lancer", type="primary"):
            try:
                cv, Q, Kb, Qs = admix_pipeline(
                    _g("gt_filt"), _g("ind_filt"),
                    range(int(k_min), int(k_max)+1), n_iter=int(n_iter))
                _s("cv_res", cv); _s("admix_Q", Q); _s("admix_K", Kb); _s("admix_all", Qs)
                st.success(f"✅ K best = {Kb}")
            except Exception as e: st.error(f"❌ {e}")
    if _ok(_g("cv_res")):
        st.plotly_chart(px.line(_g("cv_res"), x="K", y="CV", markers=True,
                                title="CV Admixture"), use_container_width=True)
    if _ok(_g("admix_Q")):
        ind = _g("ind_filt"); pc = "POP" if "POP" in ind.columns else ind.columns[0]
        fig, df = plot_admix(_g("admix_Q"), ind["IID"].astype(str).tolist(),
                             ind[pc].astype(str).tolist(), _g("admix_K"))
        st.plotly_chart(fig, use_container_width=True)

def page_ld():
    st.header("🔗 LD & ROH")
    if not _need("gt_filt"): return
    t1, t2 = st.tabs(["LD", "ROH"])
    with t1:
        max_kb = st.number_input("Fenêtre max (kb)", 10, 10000, 500)
        if st.button("▶️ LD", type="primary"):
            try:
                _s("ld_df", ld_decay(_g("gt_filt"), _g("snp_filt"), int(max_kb)))
                st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if _ok(_g("ld_df")):
            f = plot_ld(_g("ld_df"))
            if f: st.plotly_chart(f, use_container_width=True)
    with t2:
        mn = st.number_input("Nb SNPs min/ROH", 5, 500, 20)
        if st.button("▶️ ROH", type="primary"):
            try:
                roh, fr = roh_calc(_g("gt_filt"), _g("snp_filt"),
                                    _g("ind_filt"), int(mn))
                _s("roh_df", roh); _s("froh", fr)
                st.success(f"✅ {len(roh)} segments")
            except Exception as e: st.error(f"❌ {e}")
        if _ok(_g("froh")):
            st.plotly_chart(px.histogram(_g("froh"), x="FROH", nbins=30,
                                         title="FROH"), use_container_width=True)
        if _ok(_g("roh_df")):
            st.dataframe(_g("roh_df").head(50), use_container_width=True)

def page_ne():
    st.header("📉 Ne & Distances")
    if not _need("gt_filt"): return
    t1, t2 = st.tabs(["Ne", "Reynolds"])
    with t1:
        if st.button("▶️ Estimer Ne", type="primary"):
            try:
                _s("ne_dict", ne_calc(_g("ld_df"), _g("gt_filt"), _g("snp_filt")))
                st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if _ok(_g("ne_dict")):
            st.plotly_chart(px.line(_g("ne_dict"), x="generations", y="Ne",
                                    markers=True, title="Ne historique"),
                            use_container_width=True)
    with t2:
        if st.button("▶️ Reynolds", type="primary"):
            try:
                D, pops, Z, supp = reynolds_calc(_g("gt_filt"), _g("ind_filt"))
                _s("reyn_D", D); _s("reyn_pops", pops)
                _s("tree_Z", Z); _s("tree_support", supp)
                st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if _ok(_g("reyn_D")):
            st.plotly_chart(px.imshow(_g("reyn_D"), x=_g("reyn_pops"),
                                       y=_g("reyn_pops"),
                                       color_continuous_scale="Viridis",
                                       title="Reynolds"),
                            use_container_width=True)

def page_tree():
    st.header("🌳 Arbre & Réseau")
    t1, t2 = st.tabs(["Arbre NJ", "STRING"])
    with t1:
        if _ok(_g("tree_Z")):
            st.plotly_chart(plot_tree(_g("tree_Z"), _g("reyn_pops"),
                                       support=_g("tree_support"),
                                       title="Arbre NJ"),
                            use_container_width=True)
        else: st.info("Calculer Reynolds d'abord.")
    with t2:
        genes_str = st.text_area("Gènes (virgule ou newline)",
                                  "BRCA1, BRCA2, TP53, ATM, CHEK2")
        score = st.slider("Score", 150, 900, 400, 50)
        if st.button("▶️ STRING", type="primary"):
            genes = [g.strip() for g in genes_str.replace(",", "\n").split("\n") if g.strip()]
            try:
                e, n = query_string_network(genes, _g("taxon") or 9606, score)
                _s("string_edges", e); _s("string_nodes", n)
                st.success(f"✅ {len(e)} interactions")
            except Exception as ex: st.error(f"❌ {ex}")
        if _ok(_g("string_edges")):
            fig = plot_string_network(_g("string_edges"), _g("string_nodes"))
            if fig: st.plotly_chart(fig, use_container_width=True)

def page_gwas():
    st.header("🎯 GWAS")
    if not _need("gt_filt"): return
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ GWAS pairwise", type="primary"):
            try:
                ind = _g("ind_filt")
                pc = "POP" if "POP" in ind.columns else ind.columns[0]
                labels = ind[pc].astype(str).values
                uniq = sorted(set(labels))
                if len(uniq) < 2: st.error("❌ 2 populations requises")
                else:
                    res = gwas_pairwise(_g("gt_filt"), _g("snp_filt"),
                                        labels, uniq[0], uniq[1])
                    _s("gwas_res", res); _s("gwas_pheno", f"{uniq[0]} vs {uniq[1]}")
                    st.success(f"✅ {len(res)} SNPs")
            except Exception as e: st.error(f"❌ {e}")
    with c2:
        n_perm = st.number_input("Permutations", 10, 1000, 100)
        if st.button("🔁 Permutations"):
            try:
                ind = _g("ind_filt")
                pc = "POP" if "POP" in ind.columns else ind.columns[0]
                labels = ind[pc].astype(str).values
                uniq = sorted(set(labels))
                orig, p_emp, nobs, dist = permutation_test_race(
                    _g("gt_filt"), _g("snp_filt"), labels,
                    uniq[0], uniq[1], n_perm=int(n_perm))
                _s("gwas_res", orig)
                st.metric("P empirique", f"{p_emp:.4g}")
                st.metric("SNPs significatifs", nobs)
                if dist:
                    st.plotly_chart(px.histogram(x=dist, nbins=20,
                                                  title="Distribution nulle"),
                                    use_container_width=True)
            except Exception as e: st.error(f"❌ {e}")
    if _ok(_g("gwas_res")):
        st.caption(f"Phénotype : {_g('gwas_pheno', '')}")
        thr = -np.log10(0.05/max(len(_g("gwas_res")), 1))
        fig = plot_manhattan(_g("gwas_res"), "P", "Manhattan", thr=thr)
        if fig: st.plotly_chart(fig, use_container_width=True)
        st.dataframe(_g("gwas_res").head(50), use_container_width=True)

def page_meta():
    st.header("🔀 Méta-analyse Fisher")
    r = _g("gwas_res")
    if not _ok(r):
        st.warning("⚠️ Aucun GWAS en mémoire.")
        return
    if st.button("▶️ Combiner (Fisher)"):
        try:
            m = combine_pvalues_fisher(r, r)
            if m is not None:
                _s("meta_res", m); st.success(f"✅ {len(m)} SNPs")
        except Exception as e: st.error(f"❌ {e}")
    if _ok(_g("meta_res")):
        st.dataframe(_g("meta_res").head(30), use_container_width=True)

def page_vep():
    st.header("🔬 VEP & Annotation")
    if not _need("snp_filt"): return
    snp = _g("snp_filt")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### 🎯 Source des SNPs")
        source = st.radio("SNPs à annoter",
                          ["Tous les SNPs filtrés", "Top GWAS"],
                          index=0)
        top_n = 100
        if source == "Top GWAS":
            if _ok(_g("gwas_res")):
                top_n = st.number_input("Nb top SNPs", 10, 5000, 100)
            else:
                st.warning("⚠️ Aucun GWAS en mémoire — fallback sur tous les SNPs.")
                source = "Tous les SNPs filtrés"

    with c2:
        st.markdown("### ⚙️ Options VEP")
        use_vep_api = st.checkbox("Utiliser l'API Ensembl VEP (réseau requis)",
                                   value=False,
                                   help="Sinon annotation locale basique à partir du GFF")
        gff_state = "✅ GFF chargé" if _ok(_g("gff")) else "⚠️ GFF non chargé"
        st.caption(gff_state)
        if st.button("▶️ Annoter", type="primary"):
            try:
                if source == "Top GWAS":
                    gr = _g("gwas_res").sort_values("P").head(int(top_n))
                    targets = gr["SNP"].tolist()
                else:
                    targets = snp["SNP"].tolist()

                sub = snp[snp["SNP"].isin(targets)]
                annot = vep_annotate(sub, _g("species") or "human",
                                      use_api=use_vep_api, gff=_g("gff"))
                _s("snp_annot", annot)
                _s("vep_res", annot)
                st.success(f"✅ {len(annot)} SNPs annotés")
            except Exception as e:
                st.error(f"❌ {e}")

    if _ok(_g("snp_annot")):
        ann = _g("snp_annot")
        st.divider()
        st.subheader("📋 Résultats d'annotation")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SNPs annotés", len(ann))
        if "Gene" in ann.columns:
            c2.metric("Gènes uniques", ann["Gene"].dropna().nunique())
        if "Consequence" in ann.columns:
            c3.metric("Types de conséquences", ann["Consequence"].nunique())
        if "Impact" in ann.columns:
            high = (ann["Impact"] == "HIGH").sum()
            c4.metric("Impact HIGH", int(high))

        st.dataframe(ann.head(100), use_container_width=True)

        t1, t2, t3 = st.tabs(["Conséquences", "Impact", "Par chromosome"])
        with t1:
            if "Consequence" in ann.columns:
                cc = ann["Consequence"].value_counts().head(20).reset_index()
                cc.columns = ["Consequence", "N"]
                st.plotly_chart(px.bar(cc, x="N", y="Consequence", orientation="h",
                                       title="Top 20 conséquences",
                                       color="N", color_continuous_scale="Viridis"),
                                use_container_width=True)
        with t2:
            if "Impact" in ann.columns:
                ic = ann["Impact"].value_counts().reset_index()
                ic.columns = ["Impact", "N"]
                st.plotly_chart(px.pie(ic, names="Impact", values="N",
                                       title="Impact fonctionnel", hole=0.4),
                                use_container_width=True)
        with t3:
            if "CHR" in ann.columns:
                chrc = ann["CHR"].astype(str).value_counts().reset_index()
                chrc.columns = ["CHR", "N"]
                st.plotly_chart(px.bar(chrc, x="CHR", y="N",
                                       title="SNPs annotés par chromosome"),
                                use_container_width=True)

        st.download_button("⬇️ CSV annotation",
                            ann.to_csv(index=False).encode(),
                            "vep_annotation.csv", "text/csv")

def page_enrich():
    st.header("🧩 Enrichissement fonctionnel")
    if not _ok(_g("snp_annot")):
        st.warning("⚠️ Annoter les SNPs d'abord (page VEP).")
        return

    ann = _g("snp_annot")
    genes = ann["Gene"].dropna().unique().tolist() if "Gene" in ann.columns else []

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎯 Gènes candidats")
        st.metric("Gènes uniques", len(genes))
        if genes:
            st.caption(", ".join(genes[:15]) + ("..." if len(genes) > 15 else ""))
    with c2:
        st.markdown("### ⚙️ Options")
        db = st.selectbox("Base de données",
                          ["GO_BP", "GO_MF", "GO_CC", "KEGG", "Reactome"])
        p_thr = st.number_input("Seuil p-value", 0.0, 1.0, 0.05, format="%.3f")
        if st.button("▶️ Enrichissement", type="primary"):
            if not genes:
                st.error("❌ Aucun gène à analyser.")
            else:
                try:
                    res = enrich_analysis(genes, database=db, p_threshold=p_thr)
                    _s("enrich_res", res)
                    st.success(f"✅ {len(res)} termes enrichis")
                except Exception as e:
                    st.error(f"❌ {e}")

    if _ok(_g("enrich_res")):
        er = _g("enrich_res").copy()
        if "p_value" in er.columns:
            er["log_p"] = -np.log10(np.clip(er["p_value"], 1e-300, 1))
        st.divider()
        st.subheader("📋 Termes enrichis")
        st.dataframe(er.head(50), use_container_width=True)

        if "Term" in er.columns and "log_p" in er.columns:
            top = er.sort_values("log_p", ascending=False).head(20)
            fig = px.bar(top.iloc[::-1], x="log_p", y="Term", orientation="h",
                         title="Top 20 termes (‑log10 p)",
                         color="log_p", color_continuous_scale="Viridis")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇️ CSV enrichissement",
                            _g("enrich_res").to_csv(index=False).encode(),
                            "enrichissement.csv", "text/csv")

def page_gblup():
    st.header("🏗️ GBLUP")
    if not _need("gt_filt"): return

    st.markdown("### 📋 Phénotypes")
    up = st.file_uploader("Fichier phénotypes (CSV/TSV : IID, Phenotype)",
                          type=["csv", "tsv", "txt"])
    if up is not None:
        try:
            sep = "," if up.name.lower().endswith(".csv") else "\t"
            df = pd.read_csv(up, sep=sep)
            if "IID" not in df.columns or "Phenotype" not in df.columns:
                st.error("❌ Le fichier doit contenir les colonnes IID et Phenotype.")
            else:
                # FIX #5 : variable intermédiaire, plus de len(_s(...))
                df_clean = df[["IID", "Phenotype"]].dropna()
                _s("pheno_df", df_clean)
                st.success(f"✅ {len(df_clean)} phénotypes chargés")
                st.dataframe(df_clean.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ {e}")

    if not _ok(_g("pheno_df")) and _ok(_g("ind_filt")):
        ind = _g("ind_filt")
        if "Phenotype" in ind.columns and ind["Phenotype"].notna().any():
            _s("pheno_df", ind[["IID", "Phenotype"]].dropna())
            st.info("ℹ️ Phénotypes détectés dans les données individuelles.")

    if _ok(_g("pheno_df")):
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            h2 = st.slider("Héritabilité h²", 0.05, 0.95, 0.50, 0.05)
        with c2:
            model = st.selectbox("Modèle", ["GBLUP", "GBLUP + PCs"])
        with c3:
            n_pc_cov = st.number_input("Nb PCs (covariables)", 0, 20, 3,
                                        disabled=(model == "GBLUP"))

        if st.button("▶️ Lancer GBLUP", type="primary"):
            try:
                res = gblup_calc(_g("gt_filt"), _g("ind_filt"),
                                  _g("pheno_df"), h2=float(h2),
                                  n_pc=int(n_pc_cov) if model == "GBLUP + PCs" else 0)
                _s("gblup_df", res)
                st.success(f"✅ {len(res)} individus prédits")
            except Exception as e:
                st.error(f"❌ {e}")

    if _ok(_g("gblup_df")):
        g = _g("gblup_df")
        st.divider()
        st.subheader("📊 Résultats")

        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", len(g))
        if "Phenotype" in g.columns and "GEBV" in g.columns:
            corr = g[["Phenotype", "GEBV"]].corr().iloc[0, 1]
            c2.metric("r (Phénotype, GEBV)", f"{corr:.3f}")
            rmse = float(np.sqrt(np.mean((g["Phenotype"] - g["GEBV"]) ** 2)))
            c3.metric("RMSE", f"{rmse:.4f}")

        st.plotly_chart(plot_gblup(g), use_container_width=True)

        t1, t2 = st.tabs(["Tableau GEBV", "Top / Bottom"])
        with t1:
            st.dataframe(g.head(100), use_container_width=True)
        with t2:
            if "GEBV" in g.columns:
                top = g.nlargest(10, "GEBV")[["IID", "GEBV"]]
                bot = g.nsmallest(10, "GEBV")[["IID", "GEBV"]]
                st.markdown("**🏆 Top 10 GEBV**")
                st.dataframe(top, use_container_width=True)
                st.markdown("**🔻 Bottom 10 GEBV**")
                st.dataframe(bot, use_container_width=True)

        st.download_button("⬇️ CSV GEBV",
                            g.to_csv(index=False).encode(),
                            "gebv.csv", "text/csv")

def page_export():
    st.header("📤 Export")

    exports = {
        "gt_filt":   ("Génotypes filtrés (matrice)",   lambda d: pd.DataFrame(d).to_csv()),
        "snp_filt":  ("SNPs filtrés",                   lambda d: d.to_csv(index=False)),
        "ind_filt":  ("Individus filtrés",              lambda d: d.to_csv(index=False)),
        "qc_stats":  ("Stats QC",                       lambda d: pd.DataFrame([{
                          k: v for k, v in d.items()
                          if not isinstance(v, (list, tuple, np.ndarray, pd.Series))
                      }]).to_csv(index=False)),
        "pca_s":     ("Scores PCA",                     lambda d: pd.DataFrame(d).to_csv(index=False)),
        "mds_c":     ("Coordonnées MDS",                lambda d: pd.DataFrame(d).to_csv(index=False)),
        "admix_Q":   ("Q Admixture",                    lambda d: pd.DataFrame(d).to_csv(index=False)),
        "ld_df":     ("LD decay",                       lambda d: d.to_csv(index=False)),
        "roh_df":    ("Segments ROH",                   lambda d: d.to_csv(index=False)),
        "froh":      ("FROH par individu",              lambda d: d.to_csv(index=False)),
        "ne_dict":   ("Ne historique",                  lambda d: d.to_csv(index=False)),
        "reyn_D":    ("Distances Reynolds",             lambda d: pd.DataFrame(d).to_csv()),
        "gwas_res":  ("Résultats GWAS",                 lambda d: d.to_csv(index=False)),
        "meta_res":  ("Méta-analyse Fisher",            lambda d: d.to_csv(index=False)),
        "vep_res":   ("Annotation VEP",                 lambda d: d.to_csv(index=False)),
        "enrich_res":("Enrichissement fonctionnel",     lambda d: d.to_csv(index=False)),
        "gblup_df":  ("GBLUP / GEBV",                   lambda d: d.to_csv(index=False)),
        "div_stats": ("Diversité globale",              lambda d: pd.DataFrame([{
                          k: v for k, v in d.items()
                          if not isinstance(v, (list, tuple, np.ndarray, pd.Series))
                      }]).to_csv(index=False)),
        "div_pop":   ("Diversité par population",       lambda d: d.to_csv(index=False)),
        "string_edges": ("Réseau STRING — arêtes",      lambda d: d.to_csv(index=False)),
        "string_nodes": ("Réseau STRING — nœuds",       lambda d: d.to_csv(index=False)),
    }

    available = [(k, label, fn) for k, (label, fn) in exports.items()
                 if _ok(_g(k))]

    if not available:
        st.info("ℹ️ Aucun résultat disponible. Lancez d'abord des analyses.")
        return

    st.markdown(f"**{len(available)} résultat(s) prêt(s) à l'export**")

    cats = {
        "📥 Données":       ["gt_filt", "snp_filt", "ind_filt"],
        "🧪 QC & Diversité": ["qc_stats", "div_stats", "div_pop"],
        "📊 Structure":      ["pca_s", "mds_c", "admix_Q", "reyn_D"],
        "🔗 LD & Déséquilibre": ["ld_df", "roh_df", "froh", "ne_dict"],
        "🎯 Association":    ["gwas_res", "meta_res"],
        "🔬 Annotation":     ["vep_res", "enrich_res"],
        "🏗️ Prédiction":     ["gblup_df"],
        "🌐 Réseau":         ["string_edges", "string_nodes"],
    }
    by_key = {k: (label, fn) for k, label, fn in available}

    for cat, keys in cats.items():
        sub = [(k, *by_key[k]) for k in keys if k in by_key]
        if not sub:
            continue
        st.subheader(cat)
        for k, label, fn in sub:
            with st.expander(f"📄 {label}"):
                try:
                    payload = fn(_g(k))
                    if isinstance(payload, bytes):
                        payload = payload.decode()
                    st.download_button(
                        f"⬇️ Télécharger {label}.csv",
                        payload.encode() if isinstance(payload, str) else payload,
                        f"{k}.csv", "text/csv", key=f"dl_{k}")
                    if k in ["snp_filt", "ind_filt", "gwas_res", "vep_res",
                             "enrich_res", "gblup_df", "div_pop"]:
                        st.dataframe(_g(k).head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ {e}")

    st.divider()
    st.subheader("📦 Archive complète")
    if st.button("📦 Préparer archive ZIP"):
        try:
            import zipfile, io
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for k, label, fn in available:
                    try:
                        payload = fn(_g(k))
                        if isinstance(payload, bytes):
                            payload = payload.decode()
                        zf.writestr(f"{k}.csv", payload)
                    except Exception:
                        pass
                readme = (
                    "GenoScope — Export complet\n"
                    f"Espèce     : {_g('species', 'N/A')}\n"
                    f"Taxon NCBI : {_g('taxon', 'N/A')}\n"
                    f"Date       : {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n"
                    f"Résultats  : {len(available)}\n"
                )
                zf.writestr("README.txt", readme)
            buf.seek(0)
            st.download_button(
                "⬇️ Télécharger archive ZIP",
                buf.getvalue(),
                f"genoscope_{pd.Timestamp.now():%Y%m%d_%H%M}.zip",
                "application/zip")
        except Exception as e:
            st.error(f"❌ {e}")

# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(
        page_title="GenoScope",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    page = render_sidebar()

    PAGES = {
        "load":   page_load,
        "qc":     page_qc,
        "div":    page_diversity,
        "pca":    page_pca,
        "admix":  page_admix,
        "ld":     page_ld,
        "ne":     page_ne,
        "tree":   page_tree,
        "gwas":   page_gwas,
        "meta":   page_meta,
        "vep":    page_vep,
        "enrich": page_enrich,
        "gblup":  page_gblup,
        "export": page_export,
    }

    fn = PAGES.get(page)
    if fn is None:
        st.error(f"❌ Page inconnue : {page}")
        return

    try:
        fn()
    except Exception as e:
        st.error(f"❌ Erreur page « {page} » : {e}")
        with st.expander("🔎 Détail"):
            import traceback
            st.code(traceback.format_exc())

    st.sidebar.divider()
    st.sidebar.caption("🧬 GenoScope v1.0 — Partie 3/3")

if __name__ == "__main__":
    main()
