import numpy as np
import scipy.stats

# Rec. ITU-T P.1401
def or_ci(OR, N, conf):
    assert N >= 30, 'use t-Student if N<30 (not implemented)'
    alpha = 1 - conf
    sigma_or = np.sqrt(OR * (1 - OR) / N)
    z = scipy.stats.norm.ppf(1 - alpha / 2)
    
    return np.array([OR - z * sigma_or, OR + z * sigma_or])

# Rec. ITU-T P.1401
def rmse_ci(rmse, N, conf, d=4):
    alpha = 1 - conf
    n = N - d
    sqrt_n = np.sqrt(n)
    chi_alpha2 = scipy.stats.chi2.ppf(alpha / 2, n)
    chi_1malpha2 = scipy.stats.chi2.ppf(1 - alpha / 2, n)

    return np.array([(rmse * sqrt_n) / np.sqrt(chi_1malpha2), (rmse * sqrt_n) / np.sqrt(chi_alpha2)])

def corr_ci(r, N, conf):
    z_r = np.arctanh(r)
    alpha = 1 - conf
    z_alpha = scipy.stats.norm.ppf(1 - alpha / 2) * np.sqrt(1 / (N-3))

    z_L = z_r - z_alpha
    z_U = z_r + z_alpha
    
    return np.array([np.tanh(z_L), np.tanh(z_U)])

def compare_corr_max_conf(mos, mth1, mth2, mode='spearman', conf=[0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]):
    N = len(mos)
    corrf = scipy.stats.spearmanr if mode == 'spearman' else scipy.stats.pearsonr
    r01 = corrf(mth1, mos)[0]
    r02 = corrf(mth2, mos)[0]
    r12 = corrf(mth1, mth2)[0]
    
    results = [{'result': compare_corr_direct(r01, r02, r12, N, c), 'conf': c} for c in conf]
    sig_results = [x for x in results if x['result']['diff'] != 0]
    if len(sig_results) == 0:
        return 0
    else:
        return sig_results[-1]['conf']
         

# G. Y. Zou, “Toward using confidence intervals to compare correlations.,” Psychological Methods, vol. 12, no. 4, pp. 399–413, Dec. 2007, doi: 10.1037/1082-989X.12.4.399.
def compare_corr(mos, mth1, mth2, mode='spearman', conf=0.95):
    N = len(mos)
    corrf = scipy.stats.spearmanr if mode == 'spearman' else scipy.stats.pearsonr
    r01 = corrf(mth1, mos)[0]
    r02 = corrf(mth2, mos)[0]
    r12 = corrf(mth1, mth2)[0]
    
    return compare_corr_direct(r01, r02, r12, N, conf)
    
def compare_corr_direct(r01, r02, r12, N, conf=0.95):
    c = ((r12 - r01*r02/2)*(1 - r01**2 - r02**2 - r12**2) + r12**3) / ((1 - r01**2)*(1 - r02**2))
    l1, u1 = corr_ci(r01, N, conf)
    l2, u2 = corr_ci(r02, N, conf)
    
    L = r01 - r02 - np.sqrt((r01 - l1)**2 + (u2 - r02)**2 - 2*c*(r01 - l1)*(u2 - r02))
    U = r01 - r02 + np.sqrt((u1 - r01)**2 + (r02 - l2)**2 - 2*c*(u1 - r01)*(r02 - l2))
    
    assert L <= (r01 - r02) <= U, f'L: {L}, r01 - r02: {r01 - r02}, U: {U}'

    if L*U < 0:
        diff = 0
    elif L > 0:
        diff = 1
    else:
        diff = -1

    return {'diff': diff, 'r01': r01, 'r02': r02, 'l1': l1, 'u1': u1, 'l2': l2, 'u2': u2, 'L': L, 'U': U}