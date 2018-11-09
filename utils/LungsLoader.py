import os
import warnings
import numpy as np
import SimpleITK as sitk

# Change cur dir to project root
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


class LungsLoader:
    """
    This class looks for scans available on the machine and offers utilities to read the mhd files and access the data.
    """

    def __init__(self, cache=False, reduce_method=None):
        """
        Initializes the lungs loader. Looks for all available scans in the machine under data/subsetX.
        :param cache: boolean. If True, scans are cached for further use.
        """
        # Load all the lung files
        self._data_folder = os.path.join(base_dir, "data")
        self._subset_folders = [os.path.join(self._data_folder, f"subset{d}") for d in range(10)]
        self.scans = {}
        for subset_f in self._subset_folders:
            if os.path.exists(subset_f):
                for f in os.listdir(subset_f):
                    if f.split(".")[-1] == "mhd":
                        self.scans[".".join(f.split(".")[:-1])] = os.path.join(subset_f, f)
            else:
                warnings.warn(f"Subset folder {subset_f[-1:]} not found in data.")
        # Initialize cache (not used if cache is False)
        self._cache = cache
        self._cache_scans = {}
        self._cache_scans_reduced = {}
        # Initialize reduce methods (to reduce the size of the scan for further use)
        self._reduce_method = reduce_method

    def get_scan_ids(self):
        """
        Function to retrieve the scan ids of every scan available on the machine under data/subsetX
        :return: The list of ids.
        """
        return list(self.scans.keys())

    def get_scan(self, scan_id):
        """
        Function to retrieve the data concerning a specific scan.
        :param scan_id: Scan id (the list is available through get_scan_ids).
        :return: ct_scan of shape (z, x, y), origin, spacing
        """
        if scan_id not in self.scans:
            raise IndexError(f"Scan id {scan_id} not found in loaded scans. Make sure the file exists on this computer")
        if self._cache:
            if scan_id not in self._cache_scans:
                self._cache_scans[scan_id] = self._load_itk(self.scans[scan_id])
            return self._cache_scans[scan_id]
        else:
            return self._load_itk(self.scans[scan_id])

    def get_scan_reduced(self, scan_id):
        """
        Function to retrieve the reduced data concerning a specific scan.
        :param scan_id: Scan id (the list is available through get_scan_ids).
        :return: ct_scan of shape (z, x, y), origin, spacing
        """
        # TODO implement that, will be similar to get_scan but must reduce the data
        # TODO NB we must put the reducing in the data utility and not here
        raise NotImplementedError

    @staticmethod
    def _load_itk(filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        origin = np.array(list(reversed(itkimage.GetOrigin())))

        # Read the spacing along each dimension
        spacing = np.array(list(reversed(itkimage.GetSpacing())))

        return ct_scan, origin, spacing
