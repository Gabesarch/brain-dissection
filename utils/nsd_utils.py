# Author: Aria Wang
import json
import pickle
import re
from math import sqrt

import numpy as np
import torch

import ipdb
st = ipdb.set_trace

from sklearn.decomposition import PCA
import os

import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cortex

def zero_strip(s):
    if s[0] == "0":
        s = s[1:]
        return zero_strip(s)
    else:
        return s


def zscore(mat, axis=None):
    if axis is None:
        return (mat - np.mean(mat)) / np.std(mat)
    else:
        return (mat - np.mean(mat, axis=axis, keepdims=True)) / np.std(
            mat, axis=axis, keepdims=True
        )


def pearson_corr(X, Y, rowvar=True):
    if rowvar:
        return np.mean(zscore(X, axis=1) * zscore(Y, axis=1), axis=1)
    else:
        return np.mean(zscore(X, axis=0) * zscore(Y, axis=0), axis=0)

def rsq_zscore(y_true, y_pred, z_score=True, weight=1.0, rowvar=True):
    '''
    X is labels
    Y is predicted
    formula from: https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/metrics/_regression.py#L702
    '''
    if z_score:
        y_true = zscore(y_true, axis=0)
        y_pred = zscore(y_pred, axis=0)

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (
        weight * (y_true - np.average(y_true, axis=0, weights=None)) ** 2
    ).sum(axis=0, dtype=np.float64)
    output_scores = 1 - (numerator / denominator)

    return output_scores

    # if rowvar:
    #     X_zscore = zscore(X, axis=1)
    #     Y_zscore = zscore(Y, axis=1)
    #     return 1 - np.sum((X_zscore - Y_zscore)**2, axis=1) / np.sum((X_zscore - np.mean(X_zscore))**2, axis=1)
    # else:
    #     X_zscore = zscore(X, axis=0)
    #     Y_zscore = zscore(Y, axis=0)
    #     st()
    #     return 1 - (np.sum((X_zscore - Y_zscore)**2, axis=0) / np.sum((X_zscore - np.mean(X_zscore))**2, axis=0))

def empirical_p(acc, dist, dim=2):
    # dist is permute times x num_voxels
    # acc is of length num_voxels
    if dim == 1:
        return np.sum(dist > acc) / dist.shape[0]
    elif dim == 2:
        assert len(acc) == dist.shape[1]
        ps = list()
        for i, r in enumerate(acc):
            ps.append(np.sum(dist[:, i] > r) / dist.shape[0])
        return ps


def pool_size(fm, dim):
    """
    pool_size() calculates what size avgpool needs to do to reduce the 2d feature into
    desired dimension.
    :param fm: 2D feature/data matrix
    :param dim:
    :param adaptive:
    :return:
    """

    k = 1
    tot = torch.numel(torch.Tensor(fm.view(-1).shape))
    print(tot)
    ctot = tot
    while ctot > dim:
        k += 1
        ctot = tot / k / k
    return k


def check_nans(data, clean=False):
    if np.sum(np.isnan(data)) > 0:
        print("NaNs in the data")
        if clean:
            nan_sum = np.sum(np.isnan(data), axis=1)
            new_data = data[nan_sum < 1, :]
            print("Original data shape is " + data.shape)
            print("NaN free data shape is " + new_data.shape)
            return new_data
    else:
        return data


def pytorch_pca(x):
    x_mu = x.mean(dim=0, keepdim=True)
    x = x - x_mu

    _, s, v = x.svd()
    s = s.unsqueeze(0)
    nsqrt = sqrt(x.shape[0] - 1)
    xp = x @ (v / s * nsqrt)

    return xp


def pca_test(x):
    from sklearn.decomposition import PCA

    pca = PCA(whiten=True, svd_solver="full")
    pca.fit(x)
    xp = pca.transform(x)
    return xp


def sum_squared_error(x1, x2):
    return np.sum((x1 - x2) ** 2, axis=0)


def ev(data, biascorr=True):
    """Computes the amount of variance in a voxel's response that can be explained by the
    mean response of that voxel over multiple repetitions of the same stimulus.

    If [biascorr], the explainable variance is corrected for bias, and will have mean zero
    for random datasets.

    Data is assumed to be a 2D matrix: time x repeats.
    """
    ev = 1 - (data.T - data.mean(1)).var() / data.var()
    if biascorr:
        return ev - ((1 - ev) / (data.shape[1] - 1.0))
    else:
        return ev


def generate_rdm(mat, idx=None, avg=False):
    """
    Generate rdm based on data selected by the idx
    idx: lists of index if averaging is not needed; list of list of index if averaging is needed
    """
    from scipy.spatial.distance import pdist, squareform

    if idx is None:
        idx = np.arange(mat.shape[0])

    if type(mat) == list:
        data = np.array(mat)[idx]
        return np.corrcoef(data)
    if avg:
        data = np.zeros((len(idx), mat.shape[1]))
        for i in range(len(idx)):
            data[i] = np.mean(mat[idx[i], :], axis=0)
    else:
        data = mat[idx, :]

    dist = squareform(pdist(data, "cosine"))
    return dist


def negative_tail_fdr_threshold(x, chance_level, alpha=0.05, axis=-1):
    """
    The idea of this is to assume that the noise distribution around the known chance level is symmetric. We can then
    estimate how many of the values at a given level above the chance level are due to noise based on how many values
    there are at the symmetric below chance level.
    Args:
        x: The data
        chance_level: The known chance level for this metric.
            For example, if the metric is correlation, this could be 0.
        alpha: Significance level
        axis: Which axis contains the distribution of values
    Returns:
        The threshold at which only alpha of the values are due to noise, according to this estimation method
    """
    noise_values = np.where(x <= chance_level, x, np.inf)
    # sort ascending, i.e. from most extreme to least extreme
    noise_values = np.sort(noise_values, axis=axis)
    noise_values = np.where(np.isfinite(noise_values), noise_values, np.nan)

    mixed_values = np.where(x > chance_level, x, -np.inf)
    # sort descending, i.e. from most extreme to least extreme
    mixed_values = np.sort(-mixed_values, axis=axis)
    mixed_values = np.where(np.isfinite(mixed_values), mixed_values, np.nan)

    # arange gives the number of values which are more extreme in a sorted array
    num_more_extreme = np.arange(x.shape[axis])
    # if we take these to be the mixed counts, then multiplying by alpha (after including the value itself)
    # gives us the maximum noise counts, which we can use as an index
    # we also add 1 at the end to include the item at that level
    noise_counts = np.ceil(alpha * (num_more_extreme + 1)).astype(np.intp) + 1

    # filter out illegal indexes
    indicator_valid = noise_counts < noise_values.shape[axis]

    noise_values_at_counts = np.take(
        noise_values, noise_counts[indicator_valid], axis=axis
    )
    mixed_values_at_counts = np.take(
        mixed_values, np.arange(mixed_values.shape[axis])[indicator_valid], axis=axis
    )

    # if the (abs) mixed value is greater than the (abs) noise value, we would have to move to the left on the noise
    # counts to get to the mixed value (i.e. the threshold), which is in the direction of decreasing counts. Therefore
    # at this threshold, the fdr is less than alpha
    noise_values_at_counts = np.abs(noise_values_at_counts - chance_level)
    mixed_values_at_counts = np.abs(mixed_values_at_counts - chance_level)
    thresholds = np.where(
        mixed_values_at_counts >= noise_values_at_counts, mixed_values_at_counts, np.nan
    )
    # take the minimum value where this holds
    thresholds = np.nanmin(thresholds, axis=axis)
    return thresholds

def r_to_z(r):
    return np.log((1 + r) / (1 - r)) / 2.0

def z_to_r(z):
    e = np.exp(2 * z)
    return((e - 1) / (e + 1))

def r_confidence_interval(r, alpha, n):
    z = r_to_z(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha/2)  # 2-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    # Return a sequence
    return (z_to_r(lo), z_to_r(hi))

def extract_single_roi(roi_name, roi_name_dict, roi_dir, subj):
    output_masks, roi_labels = list(), list()

    if roi_name=="kastner":
        nsd_root = '' # REPLACE WITH LOCATION
        labfile_rh = os.path.join(nsd_root, 'nsddata','freesurfer','subj%02d'%subj, 'label', 'rh.Kastner2015.mgz')
        labfile_lh = os.path.join(nsd_root, 'nsddata','freesurfer','subj%02d'%subj, 'label', 'lh.Kastner2015.mgz')
        labs_rh = np.squeeze(nib.load(labfile_rh).get_fdata())
        labs_lh = np.squeeze(nib.load(labfile_lh).get_fdata())
        labs = np.concatenate([labs_rh, labs_lh], axis=0)
        print(labs.shape)
        print(np.unique(labs))
    else:
        roi_mask = np.load(
                "%s/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                % (roi_dir, subj, subj, roi_name)
            )                    

    roi_dict = roi_name_dict[roi_name]
    for k, v in roi_dict.items():
        if k > 0:
            output_masks.append(roi_mask == k)
            roi_labels.append(v)
    return output_masks, roi_labels

def get_roi_config():
    ecc_roi_names = {
        -1: "non_cortical",
        0: "cortical",
        1: "0.5 deg",
        2: "1 deg",
        3: "2 deg",
        4: "4 deg",
        5: ">4 deg",
    }

    visual_roi_names = {
        -1: "non_cortical",
        0: "cortical",
        1: "V1v",
        2: "V1d",
        3: "V2v",
        4: "V2d",
        5: "V3v",
        6: "v3d",
        7: "h4v",
    }

    place_roi_names = {
        -1: "non_cortical",
        0: "cortical",
        1: "OPA",
        2: "PPA",
        3: "RSC",
    }

    face_roi_names = {
        -1: "non_cortical",
        0: "cortical",
        1: "OFA",
        2: "FFA-1",
        3: "FFA-2",
        # 4: "mTL-faces",
        5: "aTL-faces",
    }

    word_roi_names = {
        -1: "non_cortical",
        0: "cortical",
        1: "OWFA",
        2: "VWFA-1",
        3: "VWFA-2",
        4: "mfs-words",
    }

    stream_roi_names = {
        -1: "non_cortical",
        0: "cortical",
        1:'Early',
        2:'Midventral',
        3:'Midlateral',
        4:'Midparietal',
        5:'Ventral',
        6:'Lateral',
        7:'Parietal'
    }

    searchlight_stream_roi_names = {
        -1: "non_cortical",
        0: "cortical",
        1.0: 'Early_00', 1.05: 'Early_05', 1.1: 'Early_10', 1.15: 'Early_15', 1.2: 'Early_20', 1.25: 'Early_25', 1.3: 'Early_30', 1.35: 'Early_35', 1.4: 'Early_40', 1.45: 'Early_45', 1.5: 'Early_50', 1.55: 'Early_55', 
        2.0: 'Midventral_00', 2.05: 'Midventral_05', 
        3.0: 'Midlateral_00', 3.05: 'Midlateral_05', 
        4.0: 'Midparietal_00', 4.05: 'Midparietal_05', 
        5.0: 'Ventral_00', 5.05: 'Ventral_05', 5.1: 'Ventral_10', 5.15: 'Ventral_15', 5.2: 'Ventral_20', 5.25: 'Ventral_25', 5.3: 'Ventral_30', 5.35: 'Ventral_35', 5.4: 'Ventral_40', 5.45: 'Ventral_45', 5.5: 'Ventral_50', 5.55: 'Ventral_55', 5.6: 'Ventral_60', 5.65: 'Ventral_65', 5.7: 'Ventral_70', 5.75: 'Ventral_75', 
        6.0: 'Lateral_00', 6.05: 'Lateral_05', 6.1: 'Lateral_10', 6.15: 'Lateral_15', 6.2: 'Lateral_20', 6.25: 'Lateral_25', 6.3: 'Lateral_30', 6.35: 'Lateral_35', 6.4: 'Lateral_40', 6.45: 'Lateral_45', 6.5: 'Lateral_50', 6.55: 'Lateral_55', 6.6: 'Lateral_60', 6.65: 'Lateral_65', 6.7: 'Lateral_70', 6.75: 'Lateral_75', 
        7.0: 'Parietal_00', 7.05: 'Parietal_05', 7.1: 'Parietal_10', 7.15: 'Parietal_15', 7.2: 'Parietal_20', 7.25: 'Parietal_25', 7.3: 'Parietal_30', 7.35: 'Parietal_35'
        }

    Kastner2015_roi_names = {
        # 1: "V1v",
        # 2: "V1d",
        # 3: "V2v",
        # 4: "V2d",
        # 5: "V3v",
        # 6: "V3d",
        # 7: "hV4",
        8: "VO1",
        9: "VO2",
        10: "PHC1",
        11: "PHC2",
        12: "TO2",
        13: "TO1",
        14: "LO2",
        15: "LO1",
        16: "V3B",
        17: "V3A",
        18: "IPS0",
        19: "IPS1",
        20: "IPS2",
        21: "IPS3",
        22: "IPS4",
        23: "IPS5",
        24: "SPL1",
        25: "FEF"
    }

    # HCP_MMP1_roi_names = {
    #     2: "MST",
    #     3: "V6",
    #     23: "MT",
    #     156: "V4t",
    #     157: "FST",
    # }

    # There is some overlap in ROI names so I comment them out
    HCP_MMP1_roi_names = {
        1: 'V1_HCP', 
        2: 'MST', 
        3: 'V6', 
        4: 'V2_HCP',
        5: 'V3_HCP', 
        6: 'V4_HCP', 
        7: 'V8', 
        8: '4',
        9: '3b', 
        10: 'FEF', 
        11: 'PEF', 
        12: '55b',
        13: 'V3A_HCP', 
        14: 'RSC_HCP', 
        15: 'POS2', 
        16: 'V7',
        17: 'IPS1_HCP', 
        18: 'FFC', 
        19: 'V3B_HCP', 
        20: 'LO1_HCP',
        21: 'LO2_HCP', 
        22: 'PIT', 
        23: 'MT', 
        24: 'A1',
        25: 'PSL', 26: 'SFL', 27: 'PCV', 28: 'STV',
        29: '7Pm', 30: '7m', 31: 'POS1', 32: '23d',
        33: 'v23ab', 34: 'd23ab', 35: '31pv', 36: '5m',
        37: '5mv', 38: '23c', 39: '5L', 40: '24dd',
        41: '24dv', 42: '7AL', 43: 'SCEF', 44: '6ma',
        45: '7Am', 46: '7PL', 47: '7PC', 48: 'LIPv',
        49: 'VIP', 50: 'MIP', 51: '1', 52: '2', 53: '3a',
        54: '6d', 55: '6mp', 56: '6v', 57: 'p24pr', 58: '33pr',
        59: 'a24pr', 60: 'p32pr', 61: 'a24', 62: 'd32', 63: '8BM',
        64: 'p32', 65: '10r', 66: '47m', 67: '8Av', 68: '8Ad',
        69: '9m', 70: '8BL', 71: '9p', 72: '10d', 73: '8C',
        74: '44', 75: '45', 76: '47l', 77: 'a47r', 78: '6r',
        79: 'IFJa', 80: 'IFJp', 81: 'IFSp', 82: 'IFSa', 83: 'p9-46v',
        84: '46', 85: 'a9-46v', 86: '9-46d', 87: '9a', 88: '10v',
        89: 'a10p', 90: '10pp', 91: '11l', 92: '13l', 93: 'OFC',
        94: '47s', 95: 'LIPd', 96: '6a', 97: 'i6-8', 98: 's6-8',
        99: '43', 100: 'OP4', 101: 'OP1', 102: 'OP2-3', 103: '52',
        104: 'RI', 105: 'PFcm', 106: 'PoI2', 107: 'TA2', 108: 'FOP4',
        109: 'MI', 110: 'Pir', 111: 'AVI', 112: 'AAIC', 113: 'FOP1',
        114: 'FOP3', 115: 'FOP2', 116: 'PFt', 117: 'AIP', 118: 'EC',
        119: 'PreS', 120: 'H', 121: 'ProS', 122: 'PeEc', 123: 'STGa',
        124: 'PBelt', 125: 'A5', 126: 'PHA1', 127: 'PHA3', 128: 'STSda',
        129: 'STSdp', 130: 'STSvp', 131: 'TGd', 132: 'TE1a', 133: 'TE1p',
        134: 'TE2a', 135: 'TF', 136: 'TE2p', 137: 'PHT', 138: 'PH',
        139: 'TPOJ1', 140: 'TPOJ2', 141: 'TPOJ3', 142: 'DVT', 143: 'PGp',
        144: 'IP2', 145: 'IP1', 146: 'IP0', 147: 'PFop', 148: 'PF',
        149: 'PFm', 150: 'PGi', 151: 'PGs', 152: 'V6A', 153: 'VMV1',
        154: 'VMV3', 155: 'PHA2', 156: 'V4t', 157: 'FST', 158: 'V3CD',
        159: 'LO3', 160: 'VMV2', 161: '31pd', 162: '31a', 163: 'VVC',
        164: '25', 165: 's32', 166: 'pOFC', 167: 'PoI1', 168: 'Ig',
        169: 'FOP5', 170: 'p10p', 171: 'p47r', 172: 'TGv', 173: 'MBelt',
        174: 'LBelt', 175: 'A4', 176: 'STSva', 177: 'TE1m', 178: 'PI',
        179: 'a32pr', 180: 'p24'
        }

    # HCP_to_streams_mapping = {
    #     'Early': ['V1_HCP', 'V2_HCP', 'V3_HCP'], 
    #     'Midventral': ['V4_HCP', 'PIT'], 
    #     'Midlateral': ['LO1_HCP', 'LO2_HCP'], 
    #     'Midparietal': ['V3A_HCP', 'V3CD'], 
    #     'Ventral': ['V8', 'FFC', 'PeEc', 'PHA1', 'PHA3', 'TE2a', 'TF', 'TE2p', 'PH', 'VMV1', 'VMV3', 'PHA2', 'VMV2', 'VVC', 'TGv'], 
    #     'Lateral': ['MST', 'MT', 'STSvp', 'TE1p', 'PHT', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'PGp', 'PGi', 'PGs', 'V4t', 'FST', 'LO3'], 
    #     'Parietal': ['V7', 'IPS1_HCP', 'V3B_HCP', '7PL', 'LIPv', 'VIP', 'MIP', 'IP0', 'IP1']
    #     }

    meta_name_dict = {
        "streams": stream_roi_names,
        "searchlight_streams": searchlight_stream_roi_names,
        "floc-words": word_roi_names,
        "floc-faces": face_roi_names,
        "floc-places": place_roi_names,
        "prf-visualrois": visual_roi_names,
        "prf-eccrois": ecc_roi_names,
        "Kastner2015": Kastner2015_roi_names,
        "HCP_MMP1": HCP_MMP1_roi_names
    }

    # dict_rois = {
    #     "ecc_roi_names": ecc_roi_names, 
    #     "visual_roi_names": visual_roi_names, 
    #     "place_roi_names":place_roi_names, 
    #     "place_roi_names":place_roi_names, 
    #     "face_roi_names":face_roi_names, 
    #     "word_roi_names":word_roi_names, 
    #     "roi_name_dict":roi_name_dict
    #     }

    roi_name_to_meta_name = {}
    for key in list(meta_name_dict.keys()):
        dict_cur = meta_name_dict[key]
        for value in list(dict_cur.values()):
            roi_name_to_meta_name[value] = key
    

    return meta_name_dict, roi_name_to_meta_name




def dimensionality_reduction_feats(
    X_train, 
    X_test, 
    model_name, 
    pca_train_file_path, 
    pca_test_file_path, 
    pca_fit_path, 
    method='pca', 
    override_existing_pca=False,
    n_components=0.95,
    max_feature_size=None,
    lapack_driver="gesvd"
    ):

    if method=='random_projection':
        
        print("Running Random Projection...")
        from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
        b = X_train.shape[0]
        X_train = np.reshape(X_train, (b, -1))
        min_dim = johnson_lindenstrauss_min_dim(n_samples=X_train.shape[1], eps=0.1)
        rng = np.random.RandomState(42)
        transformer = SparseRandomProjection(random_state=rng, n_components=min_dim)
        X_train = transformer.fit_transform(X_train)
        b2 = X_test.shape[0]
        X_test = np.reshape(X_test, (b2, -1))
        X_test = transformer.transform(X_test)

    elif method=='incremental_pca':

        print("RUNNING INCREMENTAL PCA...")
        # print("Getting PCs making up 95 percent of variance.")
        b = X_train.shape[0]
        n_batches = 5
        n_components=1000
        pca = IncrementalPCA(
            copy=False,
            n_components=n_components,
            batch_size=(b // n_batches)
        )
        X_train = np.reshape(X_train, (b, -1))
        X_train = pca.fit_transform(X_train)

        b2 = X_test.shape[0]
        X_test = np.reshape(X_test, (b2, -1))
        X_test = pca.transform(X_test)
        print("done.")

        if max_feature_size is not None:
            # keep only max_feature_size components
            X_train = X_train[:,:max_feature_size]
            X_test = X_test[:,:max_feature_size]

        print("saving pca features...")
        if not os.path.isfile(pca_train_file_path) or override_existing_pca:
            print("saving ", pca_train_file_path)
            print(f"Size of saved train features: {X_train.shape}")
            np.save(pca_train_file_path, X_train)
        if not os.path.isfile(pca_test_file_path) or override_existing_pca:
            print("saving ", pca_test_file_path)
            print(f"Size of saved test features: {X_test.shape}")
            np.save(pca_test_file_path, X_test
        print('done.')

    elif method=='pca':

        print("RUNNING PCA...")
        print(f"Getting PCs with n_components={n_components}")
        b = X_train.shape[0]
        # pca = PCA(n_components=0.95, copy=False)
        pca = PCA(n_components=n_components, copy=False)
        X_train = np.reshape(X_train, (b, -1))
        X_train = pca.fit_transform(X_train)

        b2 = X_test.shape[0]
        X_test = np.reshape(X_test, (b2, -1))
        X_test = pca.transform(X_test)
        print("done.")

        if max_feature_size is not None:
            # keep only max_feature_size components
            X_train = X_train[:,:max_feature_size]
            X_test = X_test[:,:max_feature_size]

        print("saving pca features...")
        if not os.path.isfile(pca_train_file_path) or override_existing_pca:
            print("saving ", pca_train_file_path)
            print(f"Size of saved train features: {X_train.shape}")
            np.save(pca_train_file_path, X_train)
        if not os.path.isfile(pca_test_file_path) or override_existing_pca:
            print("saving ", pca_test_file_path)
            print(f"Size of saved test features: {X_test.shape}")
            np.save(pca_test_file_path, X_test)

        print('done.')

    else:
        assert(False)

    return X_train, X_test

def load_mask(subj, nsd_root=''):
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    try:
        cortical_mask = np.load(
            "%s/NSD/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy" % (nsd_root, subj, subj)
        )
    except FileNotFoundError:
        cortical_mask = np.load(
            "%s/NSD/output/voxels_masks/subj%d/old/cortical_mask_subj%02d.npy" % (nsd_root, subj, subj)
        )

    sig_mask = None

    return mask, cortical_mask, sig_mask

def project_vals_to_3d(vals, mask):
    all_vals = np.zeros(mask.shape)
    all_vals[mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)
    return all_vals

def combine_two_images_by_notwhite(img1, img2, value=0):
    img1, img2 = img1.copy(), img2.copy()
    foreground = img2[: ,: ,0] + img2[: ,: ,1] + img2[: ,: ,2] == 0
    foreground = np.repeat(np.expand_dims(foreground, axis=2), 3, axis=2)
    # if not img1.flags['WRITEABLE']:
    #     img1.setflags(write=1)
    img1[foreground] = value

    visualize=False
    if visualize:
        plt.figure(1); plt.clf()
        plt.imshow(img1)
        plt.gca().axis('off')
        plt.savefig(f'data/images/test.png')
        plt.close()
        st()
    return img1

def get_roi_contour_plot(roi_data, subj, text=None, line_width=1):
    roi_data = roi_data.copy()
    roi_data[np.isnan(roi_data)] = 0.
    mask, cortical_mask, sig_mask = load_mask(subj)
    vals_3d = project_vals_to_3d(roi_data, cortical_mask)
    vals_3d_binary = vals_3d.copy()
    vals_3d_binary[vals_3d_binary!=0] = 10

    vals_3d_binary[vals_3d_binary==0] = np.nan
    roi_volume = cortex.Volume(
            vals_3d_binary,
            "subj%02d" % subj,
            "func1pt8_to_anat0pt8_autoFSbbr",
            mask=mask,
            vmin=0,
            vmax=1, 
            cmap="Greys"
        )

    fig = cortex.quickflat.make_figure(roi_volume, with_rois=False, with_colorbar=False, nanmean=True, sampler="nearest")
    # plt.savefig(f'data/images/test.png')

    canvas = FigureCanvas(fig)
    canvas.draw()       # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    
    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image = np.ones_like(image)*255
    image = image.astype(np.uint8)
    cv2.drawContours(image, contours, -1, (0, 0, 0), line_width)

    if text is not None:
        largest_contour_idx = [0, 0]
        for cont_i in range(len(contours)):
            if len(contours[cont_i])>largest_contour_idx[1]:
                largest_contour_idx[0] = cont_i
                largest_contour_idx[1] = len(contours[cont_i])
        points = contours[largest_contour_idx[0]].squeeze(1)
        x_1, y_1 = list(np.mean(points, axis=0).astype(np.int32))
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (x_1, y_1)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    visualize=False
    if visualize:
        plt.figure(1); plt.clf()
        plt.imshow(image)
        plt.gca().axis('off')
        plt.savefig(f'data/images/test.png')
        plt.close()
        st()
    return image


if __name__ == "__main__":
    import torch
    from scipy.stats import pearsonr

    # PCA test
    x = np.array([[12.0, -51, 4, 99], [6, 167, -68, -129], [-4, 24, -41, 77]])
    x = torch.from_numpy(x).to(dtype=torch.float64)
    xp1 = pytorch_pca(x)

    xp2 = pca_test(x)
    assert np.sum(abs(xp1.numpy() - xp2) > 0.5) == 0

    # correlation test
    a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    b = np.array([[12.0, -51, 4], [99, 6, 167], [-68, -129, -4], [24, -41, 77]])

    corr_row_1 = pearson_corr(a, b, rowvar=True).astype(np.float32)
    corr_row_2 = []
    for i in range(a.shape[0]):
        corr_row_2.append(pearsonr(a[i, :], b[i, :])[0].astype(np.float32))
    assert [corr_row_1[i] == corr_row_2[i] for i in range(len(corr_row_2))]

    corr_col_1 = pearson_corr(a, b, rowvar=False).astype(np.float32)
    corr_col_2 = []
    for i in range(a.shape[1]):
        corr_col_2.append(pearsonr(a[:, i], b[:, i])[0].astype(np.float32))
    assert [corr_col_1[i] == corr_col_2[i] for i in range(len(corr_col_2))]