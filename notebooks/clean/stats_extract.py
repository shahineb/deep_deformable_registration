import sys
# from LungsLoader import LungsLoader
# from ScanHandler import ScanHandler
import numpy as np
import math
from tqdm import tqdm_notebook
sys.path.append("../../utils")
Lungs = LungsLoader()


def get_intensity(scan):
    sq = math.sqrt(np.sum(np.square(np.maximum(scan, np.zeros(scan.shape)))))
    # Add stats
    return [sq]


def get_inf(lungs, list_scans):
    """ Takes Lungs = LungsLoader class and list_scans = the list of paths of the scans we want to load
    returns the list of origins, intensities, spacing and sizes """

    origins = np.zeros((len(list_scans), 3))
    sizes = np.zeros((len(list_scans), 3))
    spacings = np.zeros((len(list_scans), 3))
    intensities = np.zeros((len(list_scans), len(get_intensity(np.zeros((300, 300))))))

    for i in tqdm_notebook(range(len(list_scans))):
        scan, origin, spacing = lungs._load_itk(list_scans[i])
        origins[i, :] = origin
        sizes[i, :] = scan.shape
        spacings[i, :] = spacing
        intensities[i, :] = get_intensity(scan)
    return origins, sizes, intensities, spacings


origins2, sizes2, intensities2, spacings2 = get_inf(Lungs, list(Lungs.scans.values()))
np.savetxt('Stats_results/origins.csv', origins2)
np.savetxt('Stats_results/sizes.csv', sizes2)
np.savetxt('Stats_results/intensities.csv', intensities2)
np.savetxt('Stats_results/spacings.csv', spacings2)