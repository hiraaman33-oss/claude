# =============================================================================
# ACACIA EVs vs NVs  |  SERS on Ag/SiNWs  |  pre-processing + PCA / LDA / PLS-DA
#
# Implements the "Data analysis" section of the manuscript
# (Acacia_EVs_NVs_AgSiNWs_Manuscript_Draft_v7):
#   per spectral window (fingerprint 200-1800, CH-stretching 2700-3100 cm-1):
#   1. cosmic-ray removal, modified z-score on first differences, threshold 7
#   2. interpolation onto a 2 cm-1 grid (205-1795 and 2705-3095 cm-1)
#   3. Savitzky-Golay smoothing, 11 points, 3rd order
#   4. asymmetric least squares baseline, lambda = 1e5, p = 0.01, 10 iterations
#   5. SNV normalisation within the window
#   analysis regions: fingerprint 250-1790, CH-stretching 2760-3095 cm-1
#
# Input: EV_Acacia_Raman_Data.xlsx, NV_Acacia_Raman_Data.xlsx
#        (one sheet per measurement site m1-m6, 20 raw spectra per site)
# =============================================================================

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

SITES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
CLASSES = ['EV', 'NV']                       # label 0 = EV, 1 = NV
WINDOWS = {
    'fingerprint': {'raw': (200, 1800), 'grid': (205, 1795), 'region': (250, 1790)},
    'ch':          {'raw': (2700, 3100), 'grid': (2705, 3095), 'region': (2760, 3095)},
}


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------
def load_workbook(path):
    """Return list of (site, wavenumber array, intensity matrix [n_px, n_wn])."""
    out = []
    for site in SITES:
        df = pd.read_excel(path, sheet_name=site, header=2)
        x = df.iloc[:, 0].to_numpy(float)
        y = df.iloc[:, 1:].to_numpy(float).T
        out.append((site, x, y))
    return out


# ----------------------------------------------------------------------------
# PRE-PROCESSING
# ----------------------------------------------------------------------------
def despike(y, threshold=7.0, m=3):
    """Whitaker & Hayes modified z-score on first differences."""
    d = np.diff(y)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad == 0:
        return y.copy()
    z = 0.6745 * (d - med) / mad
    spikes = np.zeros(len(y), bool)
    spikes[1:] = np.abs(z) > threshold
    out = y.copy()
    for i in np.where(spikes)[0]:
        lo, hi = max(0, i - m), min(len(y), i + m + 1)
        idx = [j for j in range(lo, hi) if not spikes[j]]
        if idx:
            out[i] = y[idx].mean()
    return out


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """Asymmetric least squares baseline (Eilers & Boelens)."""
    n = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    DD = lam * D.dot(D.transpose())
    w = np.ones(n)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess_window(x, Y, win):
    """Despike, interpolate, smooth, baseline. Returns grid, corrected, SNV."""
    lo, hi = WINDOWS[win]['raw']
    g0, g1 = WINDOWS[win]['grid']
    grid = np.arange(g0, g1 + 1, 2.0)
    m = (x >= lo - 5) & (x <= hi + 5)
    corr = []
    for y in Y:
        yd = despike(y[m])
        yi = np.interp(grid, x[m], yd)
        ys = savgol_filter(yi, 11, 3)
        corr.append(ys - als_baseline(ys))
    corr = np.array(corr)
    snv = (corr - corr.mean(1, keepdims=True)) / corr.std(1, keepdims=True)
    return grid, corr, snv


def build_dataset(ev_path, nv_path):
    """Pre-processed data for both windows, with class and site labels."""
    data = {w: {'corr': [], 'snv': []} for w in WINDOWS}
    labels, sites = [], []
    for cls, path in enumerate([ev_path, nv_path]):
        for site, x, Y in load_workbook(path):
            for w in WINDOWS:
                grid, corr, snv = preprocess_window(x, Y, w)
                data[w]['grid'] = grid
                data[w]['corr'].append(corr)
                data[w]['snv'].append(snv)
            labels += [cls] * len(Y)
            sites += [f'{CLASSES[cls]}-{site}'] * len(Y)
    for w in WINDOWS:
        data[w]['corr'] = np.vstack(data[w]['corr'])
        data[w]['snv'] = np.vstack(data[w]['snv'])
    return data, np.array(labels), np.array(sites)


def region(data, win):
    """Wavenumbers and SNV spectra restricted to the analysis region."""
    r0, r1 = WINDOWS[win]['region']
    g = data[win]['grid']
    m = (g >= r0) & (g <= r1)
    return g[m], data[win]['snv'][:, m]


# ----------------------------------------------------------------------------
# MULTIVARIATE ANALYSIS
# ----------------------------------------------------------------------------
def pca_fit(X, n=5):
    pca = PCA(n_components=n, svd_solver='full').fit(X)
    return pca, pca.transform(X)


def cv_folds(y, groups, n_splits=5, seed=0):
    """Site-grouped stratified folds. seed=0 reproduces the per-class counts of
    Table 1 for the fingerprint region and for CH-stretching PCA-LDA."""
    sk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sk.split(np.zeros(len(y)), y, groups))


def cv_pca_lda(X, y, groups, folds, refit=False, n=5):
    """PCA-LDA accuracy; PCA on pooled data (refit=False) or per training fold."""
    pred = np.empty(len(y), int)
    if not refit:
        T = PCA(n_components=n, svd_solver='full').fit_transform(X)
    for tr, te in folds:
        if refit:
            pca = PCA(n_components=n, svd_solver='full').fit(X[tr])
            Ttr, Tte = pca.transform(X[tr]), pca.transform(X[te])
        else:
            Ttr, Tte = T[tr], T[te]
        pred[te] = LinearDiscriminantAnalysis().fit(Ttr, y[tr]).predict(Tte)
    return pred


def cv_plsda(X, y, folds, n=5):
    pred = np.empty(len(y), int)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        pls = PLSRegression(n_components=n, scale=False).fit(sc.transform(X[tr]), y[tr])
        pred[te] = (pls.predict(sc.transform(X[te])).ravel() > 0.5).astype(int)
    return pred


def lda_scores(X, y, n=5):
    """LD1 scores of all spectra from PCA (pooled) + LDA fitted on all data."""
    T = PCA(n_components=n, svd_solver='full').fit_transform(X)
    return LinearDiscriminantAnalysis().fit(T, y).transform(T).ravel()


# ----------------------------------------------------------------------------
# BAND STATISTICS (Table S1)
# ----------------------------------------------------------------------------
TABLE_S1 = [  # (label, lo, hi, window)
    ('519', 510, 530, 'fingerprint'), ('826', 820, 835, 'fingerprint'),
    ('855', 850, 862, 'fingerprint'), ('1001', 995, 1008, 'fingerprint'),
    ('1031', 1025, 1040, 'fingerprint'), ('1160', 1150, 1175, 'fingerprint'),
    ('1245', 1235, 1255, 'fingerprint'), ('1380', 1340, 1420, 'fingerprint'),
    ('1520', 1505, 1535, 'fingerprint'), ('1580', 1570, 1590, 'fingerprint'),
    ('1605', 1598, 1615, 'fingerprint'), ('1725', 1710, 1735, 'fingerprint'),
    ('2850', 2840, 2860, 'ch'), ('2880', 2870, 2890, 'ch'), ('2930', 2920, 2940, 'ch'),
    ('2960', 2950, 2970, 'ch'), ('3060', 3050, 3075, 'ch'),
]


def site_means(values, y, sites):
    out = {}
    for c in (0, 1):
        s = sorted(set(sites[y == c]))
        out[c] = np.array([values[sites == k].mean() for k in s])
    return out


def band_table(data, y, sites):
    rows = []
    for lab, lo, hi, w in TABLE_S1:
        g = data[w]['grid']
        m = (g >= lo) & (g <= hi)
        v = data[w]['snv'][:, m].mean(1)
        sm = site_means(v, y, sites)
        p = ttest_ind(sm[0], sm[1], equal_var=False).pvalue
        rows.append(dict(band=lab, lo=lo, hi=hi, window=w,
                         ev_mean=sm[0].mean(), ev_sd=sm[0].std(ddof=1),
                         nv_mean=sm[1].mean(), nv_sd=sm[1].std(ddof=1),
                         p_raw=p, p_bonf=min(1.0, p * len(TABLE_S1))))
    return pd.DataFrame(rows)


def ch_ratio(data, y, sites):
    """I2960/I2930 from baseline-corrected spectra, as site mean +/- SD."""
    g, C = data['ch']['grid'], data['ch']['corr']
    r = C[:, (g >= 2950) & (g <= 2970)].mean(1) / C[:, (g >= 2920) & (g <= 2940)].mean(1)
    return site_means(r, y, sites)
