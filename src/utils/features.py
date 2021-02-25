import numpy as np
from .color_space import rgb_to_yuv
from numba import njit

@njit
def psnr(value, max_energy):
    if value == 0:
        return 75
    return 10 * np.log10(max_energy / value)

@njit
def mse_features_njit(ref_pc_val, ref_pc_val_ngb, deg_pc_val, deg_pc_val_ngb):
    ref_pc_val_mask = ~np.isnan(ref_pc_val) & ~np.isinf(ref_pc_val) & ~np.isnan(ref_pc_val_ngb) & ~np.isinf(ref_pc_val_ngb)
    ref_pc_val = ref_pc_val[ref_pc_val_mask]
    ref_pc_val_ngb = ref_pc_val_ngb[ref_pc_val_mask]

    deg_pc_val_mask = (~np.isnan(deg_pc_val)) & (~np.isinf(deg_pc_val)) & ~np.isnan(deg_pc_val_ngb) & ~np.isinf(deg_pc_val_ngb)
    deg_pc_val = deg_pc_val[deg_pc_val_mask]
    deg_pc_val_ngb = deg_pc_val_ngb[deg_pc_val_mask]
    
    f_AB = np.mean(np.square(ref_pc_val - ref_pc_val_ngb))
    f_BA = np.mean(np.square(deg_pc_val - deg_pc_val_ngb))
    f_max = max(f_AB, f_BA)
    
    return f_AB, f_BA, f_max
    
def mse_features(col_name, features, ref_pc, ref_pc_ngb, deg_pc, deg_pc_ngb):
    key_AB = f'{col_name}_mse_AB'
    key_BA = f'{col_name}_mse_BA'
    key_max = f'max_{col_name}_mse'
    result = mse_features_njit(ref_pc[col_name].values, ref_pc_ngb[col_name].values,
                               deg_pc[col_name].values, deg_pc_ngb[col_name].values)
    features[key_AB], features[key_BA], features[key_max] = result

def compute_features(ref_data, deg_data):
    ref_pc = ref_data['pc']
    deg_pc = deg_data['pc']
    ngb_AB = deg_data['ngb_AB']
    ngb_BA = deg_data['ngb_BA']
    ref_pc_ngb = deg_pc.iloc[ngb_AB]
    deg_pc_ngb = ref_pc.iloc[ngb_BA]
    
    xyz = ['x', 'y', 'z']
    nxyz = ['nx', 'ny', 'nz']
    ref_pc_xyz = ref_pc[xyz].values
    ref_pc_xyz_ngb = ref_pc_ngb[xyz].values
    ref_pc_nxyz_ngb = ref_pc_ngb[nxyz].values
    deg_pc_xyz = deg_pc[xyz].values
    deg_pc_xyz_ngb = deg_pc_ngb[xyz].values
    deg_pc_nxyz_ngb = deg_pc_ngb[nxyz].values
    
    max_energy = 3 * ((2 ** deg_data['geometry_bits']) ** 2)
    
    features = {}
    features['input_points'] = len(ref_pc)
    features['output_points'] = len(deg_pc)
    features['out_in_points_ratio'] = len(ref_pc) / len(deg_pc)
    
    features['d1_mse_AB'] = np.mean(np.square(deg_data['dists_AB']))
    features['d1_mse_BA'] = np.mean(np.square(deg_data['dists_BA']))
    features['max_d1_mse'] = max(features['d1_mse_AB'], features['d1_mse_BA'])
    
    features['d1_psnr_AB'] = psnr(features['d1_mse_AB'], max_energy)
    features['d1_psnr_BA'] = psnr(features['d1_mse_BA'], max_energy)
    features['min_d1_psnr'] = min(features['d1_psnr_AB'], features['d1_psnr_BA'])

    features['geometry_bits'] = deg_data['geometry_bits']
    features['max_energy'] = max_energy
    
    ref_pc_xyz_diff = ref_pc_xyz - ref_pc_xyz_ngb
    deg_pc_xyz_diff = deg_pc_xyz - deg_pc_xyz_ngb

    features['d2_mse_AB'] = np.mean(np.sum(ref_pc_xyz_diff * ref_pc_nxyz_ngb, axis=1) ** 2, axis=0)
    features['d2_mse_BA'] = np.mean(np.sum(deg_pc_xyz_diff * deg_pc_nxyz_ngb, axis=1) ** 2, axis=0)
    features['max_d2_mse'] = max(features['d2_mse_AB'], features['d2_mse_BA'])
    
    features['d2_psnr_AB'] = psnr(features['d2_mse_AB'], max_energy)
    features['d2_psnr_BA'] = psnr(features['d2_mse_BA'], max_energy)
    features['min_d2_psnr'] = min(features['d2_psnr_AB'], features['d2_psnr_BA'])
    
    rgb = ['red', 'green', 'blue']
    ref_pc_col = rgb_to_yuv(ref_pc[rgb].values)
    ref_pc_col_ngb =  rgb_to_yuv(ref_pc_ngb[rgb].values)
    deg_pc_col = rgb_to_yuv(deg_pc[rgb].values)
    deg_pc_col_ngb = rgb_to_yuv(deg_pc_ngb[rgb].values)
    
    mse_color_AB = np.mean(np.square((ref_pc_col - ref_pc_col_ngb) / 255.), axis=0)
    mse_color_BA = np.mean(np.square((deg_pc_col - deg_pc_col_ngb) / 255.), axis=0)
    psnr_color_AB = 10 * np.log10(1 / mse_color_AB)
    psnr_color_BA = 10 * np.log10(1 / mse_color_BA)
    psnr_color_AB[mse_color_AB == 0] = 50
    psnr_color_BA[mse_color_BA == 0] = 50
    for i, component in enumerate(['y', 'u', 'v']):
        features[f'{component}_mse_color_AB'] = mse_color_AB[i]
        features[f'{component}_mse_color_BA'] = mse_color_BA[i]
        features[f'max_{component}_mse_color'] = max(mse_color_AB[i], mse_color_BA[i])
        features[f'{component}_psnr_color_AB'] = psnr_color_AB[i]
        features[f'{component}_psnr_color_BA'] = psnr_color_BA[i]
        features[f'min_{component}_psnr_color'] = min(psnr_color_AB[i], psnr_color_BA[i])

    return features
