# =============================================================================
# ACACIA EVs vs NVs  |  independent verification of acacia_pipeline.py
#
# Re-computes every result with separately written code (different Excel
# reader, banded ALS solver, numpy-SVD PCA, hand-written Fisher LDA, manual
# Welch's t-test) and compares with acacia_pipeline.py. Also checks data
# integrity, fold leakage and a site-label permutation (expected ~50 %).
#
# HOW TO RUN:
#   python Acacia_verify.py  EV_Acacia_Raman_Data.xlsx  NV_Acacia_Raman_Data.xlsx
# =============================================================================

import sys
import warnings

import numpy as np
import openpyxl
from scipy.linalg import solve_banded
from scipy.signal import savgol_filter
from scipy import stats

import acacia_pipeline as ap

warnings.filterwarnings('ignore')
EV_FILE, NV_FILE = sys.argv[1], sys.argv[2]
report = []


def check(name, ok, detail=''):
    report.append(f'[{"PASS" if ok else "FAIL"}] {name}  {detail}')


# ----------------------------------------------------------------------------
# 1. DATA INTEGRITY - read with openpyxl, compare with the pandas reader
# ----------------------------------------------------------------------------
def read_openpyxl(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for site in ap.SITES:
        rows = list(wb[site].iter_rows(min_row=4, values_only=True))
        a = np.array([[np.nan if v is None else v for v in r] for r in rows], float)
        a = a[~np.isnan(a[:, 0])]
        out.append((site, a[:, 0], a[:, 1:].T))
    return out


for path, tag in [(EV_FILE, 'EV'), (NV_FILE, 'NV')]:
    A, B = read_openpyxl(path), ap.load_workbook(path)
    same = all(np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) for a, b in zip(A, B))
    check(f'{tag}: two independent Excel readers give identical data', same)
    n = sum(len(a[2]) for a in A)
    check(f'{tag}: 6 sites x 20 spectra', n == 120 and all(len(a[2]) == 20 for a in A), f'n = {n}')
    check(f'{tag}: no missing values', all(np.isfinite(a[2]).all() for a in A))
    check(f'{tag}: wavenumber axis increasing', all(np.all(np.diff(a[1]) > 0) for a in A))
    check(f'{tag}: positive counts', all((a[2] > 0).all() for a in A),
          f'min = {min(a[2].min() for a in A):.1f}')
    dup = 0
    for a in A:
        for i in range(20):
            for j in range(i + 1, 20):
                dup += np.array_equal(a[2][i], a[2][j])
    check(f'{tag}: no duplicated spectra within a site', dup == 0)


# ----------------------------------------------------------------------------
# 2. PRE-PROCESSING - independent implementation
# ----------------------------------------------------------------------------
def despike2(y, thr=7.0, m=3):
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    z = np.zeros(len(y))
    z[1:] = 0.6745 * (d - np.median(d)) / mad
    bad = np.abs(z) > thr
    out = y.copy()
    for i in np.flatnonzero(bad):
        nb = [j for j in range(max(0, i - m), min(len(y), i + m + 1)) if not bad[j]]
        out[i] = y[nb].mean()
    return out


def als2(y, lam=1e5, p=0.01, niter=10):
    """ALS with a banded (pentadiagonal) solver instead of sparse spsolve."""
    n = len(y)
    D = np.diff(np.eye(n), 2, axis=0)
    H = lam * D.T @ D
    ab = np.zeros((5, n))
    for k in range(-2, 3):
        ab[2 - k, max(0, k):n + min(0, k)] = np.diagonal(H, k)
    w = np.ones(n)
    for _ in range(niter):
        ab2 = ab.copy()
        ab2[2] += w
        z = solve_banded((2, 2), ab2, w * y)
        w = np.where(y > z, p, 1 - p)
    return z


def prep2(path):
    out = {'fingerprint': [], 'ch': []}
    for site, x, Y in read_openpyxl(path):
        for w in out:
            lo, hi = ap.WINDOWS[w]['raw']
            grid = np.arange(ap.WINDOWS[w]['grid'][0], ap.WINDOWS[w]['grid'][1] + 1, 2.0)
            m = (x >= lo - 5) & (x <= hi + 5)
            for y in Y:
                s = savgol_filter(np.interp(grid, x[m], despike2(y[m])), 11, 3)
                c = s - als2(s)
                out[w].append((c - c.mean()) / c.std())
    return {w: np.array(v) for w, v in out.items()}


data, y, sites = ap.build_dataset(EV_FILE, NV_FILE)
p_ev, p_nv = prep2(EV_FILE), prep2(NV_FILE)
for w in ['fingerprint', 'ch']:
    X2 = np.vstack([p_ev[w], p_nv[w]])
    dmax = np.abs(X2 - data[w]['snv']).max()
    check(f'{w}: pre-processing reproduced independently', dmax < 1e-6, f'max |diff| = {dmax:.1e}')


# ----------------------------------------------------------------------------
# 3. PCA (numpy SVD), LDA (hand-written Fisher), PLS-DA, leakage, permutation
# ----------------------------------------------------------------------------
def pca_svd(X, n=5):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n].T, (S ** 2 / (S ** 2).sum())[:n]


def fisher_lda_predict(Ttr, ytr, Tte):
    m0, m1 = Ttr[ytr == 0].mean(0), Ttr[ytr == 1].mean(0)
    Sw = np.cov(Ttr[ytr == 0].T) * (sum(ytr == 0) - 1) + np.cov(Ttr[ytr == 1].T) * (sum(ytr == 1) - 1)
    Sw /= len(ytr) - 2
    wv = np.linalg.solve(Sw, m1 - m0)
    prior = np.log(np.mean(ytr == 1) / np.mean(ytr == 0))
    score = (Tte - (m0 + m1) / 2) @ wv + prior
    return (score > 0).astype(int)


folds = ap.cv_folds(y, sites)
leak = any(set(sites[tr]) & set(sites[te]) for tr, te in folds)
check('no measurement site in both training and test fold', not leak)
check('every spectrum tested exactly once',
      np.array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(len(y))))

manuscript = {'fingerprint': (16.8, 8.9, 95.4, 95.4), 'ch': (33.6, 13.8, 97.1, 97.9)}
for w in ['fingerprint', 'ch']:
    g, X = ap.region(data, w)
    T, var = pca_svd(X)
    _, T1 = ap.pca_fit(X)
    pca, _ = ap.pca_fit(X)
    check(f'{w}: PCA variance, SVD = scikit-learn',
          np.allclose(var, pca.explained_variance_ratio_), f'{np.round(var[:2] * 100, 1)} %')
    pred_hand = np.empty(len(y), int)
    for tr, te in folds:
        pred_hand[te] = fisher_lda_predict(T[tr], y[tr], T[te])
    pred_ap = ap.cv_pca_lda(X, y, sites, folds)
    check(f'{w}: PCA-LDA, hand-written LDA = scikit-learn',
          np.array_equal(pred_hand, pred_ap), f'accuracy {(pred_ap == y).mean() * 100:.1f} %')
    pls = (ap.cv_plsda(X, y, folds) == y).mean() * 100
    ms = manuscript[w]
    report.append(f'       {w}: PC1/PC2 {var[0] * 100:.1f}/{var[1] * 100:.1f} % (manuscript '
                  f'{ms[0]}/{ms[1]}); PCA-LDA {(pred_ap == y).mean() * 100:.1f} % '
                  f'(manuscript {ms[2]}); PLS-DA {pls:.1f} % (manuscript {ms[3]})')
    # permutation: shuffle class labels between whole sites -> expected near 50 %
    rng = np.random.default_rng(1)
    accs = []
    site_ids = np.array(sorted(set(sites)))
    for _ in range(50):
        perm = dict(zip(site_ids, rng.permutation([0] * 6 + [1] * 6)))
        yp = np.array([perm[s] for s in sites])
        fp = ap.cv_folds(yp, sites)
        accs.append((ap.cv_pca_lda(X, yp, sites, fp) == yp).mean() * 100)
    check(f'{w}: site-label permutation gives chance-level accuracy', np.mean(accs) < 75,
          f'mean {np.mean(accs):.1f} %, max {np.max(accs):.1f} % (50 permutations)')


# ----------------------------------------------------------------------------
# 4. Table S1 - manual Welch's t-test
# ----------------------------------------------------------------------------
bt = ap.band_table(data, y, sites)
ok = True
for _, b in bt.iterrows():
    gw = data[b.window]['grid']
    v = data[b.window]['snv'][:, (gw >= b.lo) & (gw <= b.hi)].mean(1)
    a = np.array([v[sites == s].mean() for s in sorted(set(sites[y == 0]))])
    c = np.array([v[sites == s].mean() for s in sorted(set(sites[y == 1]))])
    va, vc = a.var(ddof=1) / 6, c.var(ddof=1) / 6
    t = (a.mean() - c.mean()) / np.sqrt(va + vc)
    df = (va + vc) ** 2 / (va ** 2 / 5 + vc ** 2 / 5)
    p = 2 * stats.t.sf(abs(t), df)
    ok &= np.isclose(p, b.p_raw)
check('Table S1: manual Welch t-test = scipy for all 17 bands', ok)

open('Acacia_verification_report.txt', 'w').write('\n'.join(report) + '\n')
print('\n'.join(report))
