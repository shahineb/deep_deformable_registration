import numpy as np
from sklearn.metrics import mean_squared_error, normalized_mutual_info_score
from scipy.ndimage import morphology

CONTINOUS = "CONTINOUS"
DISCRETE = "DISCRETE"


def metric(domain):
    """Decorator to specify domain metric is meant for

    Args:
        domain (str):
            - "CONTINUOUS": real valued arrays
            - "DISCRETE": labels arrays
    """
    def metric_decorator(func):
        func._domain = domain
        return func
    return metric_decorator


@metric(CONTINOUS)
def cross_correlation(vol1, vol2):
    """Cross correlation between two N-d arrays

    Args:
        vol1 (np.ndarray)
        vol2 (np.ndarray)
    """
    mean_1 = np.mean(vol1)
    mean_2 = np.mean(vol2)
    var_1 = np.sum(np.square(vol1 - mean_1))
    var_2 = np.sum(np.square(vol2 - mean_2))
    cov_12 = np.sum((vol2 - mean_2) * (vol1 - mean_1))
    return np.square(cov_12 / np.sqrt(var_1 * var_2 + 1e-5))


@metric(CONTINOUS)
def mse(vol1, vol2):
    """Mean Square Error between two N-d arrays

    Args:
        vol1 (np.ndarray)
        vol2 (np.ndarray)
    """
    return mean_squared_error(vol1.reshape(1, -1), vol2.reshape(1, -1))


def _surface_distance(seg1, seg2, sampling=1, connectivity=1):
    """Computes surface distance between to segmentation volumes as
    explained in https://mlnotebook.github.io/post/surface-distance-function/

    Args:
        seg1 (np.ndarray): integer valued labels volume
        seg2 (np.ndarray): integer valued labels volume
        sampling (np.ndarray): Pixel resolution (default: 1 -> np.array([1, 1, 1]))
        connectivity (int): Neighbouring approach, default 6-neighbours kernel
    """
    seg1 = np.atleast_1d(seg1.astype(np.bool))
    seg2 = np.atleast_1d(seg2.astype(np.bool))
    conn = morphology.generate_binary_structure(seg1.ndim, connectivity)

    S = seg1 ^ morphology.binary_erosion(seg1, conn)
    Sprime = seg2 ^ morphology.binary_erosion(seg2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])
    return sds


@metric(DISCRETE)
def haussdorf_distance(seg1, seg2):
    """Hauddsorf Distance between two segmentation volumes

    Args:
        seg1 (np.ndarray): integer valued labels volume
        seg2 (np.ndarray): integer valued labels volume
    """
    sds = _surface_distance(seg1, seg2)
    return sds.max()


@metric(DISCRETE)
def msd(seg1, seg2):
    """Mean Surface Distance between two segmentation volumes

    Args:
        seg1 (np.ndarray): integer valued labels volume
        seg2 (np.ndarray): integer valued labels volume
    """
    sds = _surface_distance(seg1, seg2)
    return sds.mean()


@metric(DISCRETE)
def rms(seg1, seg2):
    """Residual Mean Square Distance between two segmentation volumes

    Args:
        seg1 (np.ndarray): integer valued labels volume
        seg2 (np.ndarray): integer valued labels volume
    """
    sds = _surface_distance(seg1, seg2)
    return np.sqrt((sds**2).mean())


@metric(DISCRETE)
def dice_score(seg1, seg2):
    """Dice score between two segmentation volumes
    Background must be labeled 0

    Args:
        seg1 (np.ndarray): integer valued labels volume
        seg2 (np.ndarray): integer valued labels volume
    """
    numerator = 2 * np.sum(np.equal(seg1, seg2))
    denominator = seg1.size + seg2.size
    return numerator / denominator


@metric(DISCRETE)
def mutual_info(seg1, seg2):
    """Mutual information between two segmentation volumes

    Args:
        seg1 (np.ndarray): integer valued labels volume
        seg2 (np.ndarray): integer valued labels volume
    """
    return normalized_mutual_info_score(seg1.flatten(), seg2.flatten())
