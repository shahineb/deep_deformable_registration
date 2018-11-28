import os
import warnings
import numpy as np
import SimpleITK as sitk

# Change cur dir to project root
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


class LungsLoader:
    """
    This class looks for scans available on the machine and offers utilities to read the mhd files
    and access the data.
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
        self._cache_scans = {"resampled": {}, "not_resampled": {}}
        self._cache_scans_reduced = {}
        # Initialize reduce methods (to reduce the size of the scan for further use)
        self._reduce_method = reduce_method

    def get_scan_ids(self):
        """
        Function to retrieve the scan ids of every scan available on the machine under data/subsetX
        :return: The list of ids.
        """
        return list(self.scans.keys())

    @staticmethod
    def _get_itk_from_scan(ct_scan, origin, spacing):
        """
        Convert a ct_scan to a SimplteITK image.
        :param ct_scan: Original ct_scan of shape (z, x, y).
        :param origin: Origins of the ct_scan to convert.
        :param spacing: Spacings of the ct_scan to convert.
        :return: The corresponding SimpleITK image.
        """
        img = sitk.GetImageFromArray(ct_scan)
        img.SetOrigin((origin[1], origin[2], origin[0]))
        img.SetSpacing((spacing[1], spacing[2], spacing[0]))
        return img

    @staticmethod
    def _resample(ct_scan, origin, spacing, spacing_x, spacing_y, spacing_z):
        """
        Takes as input the informations about the scan to resample and the new spacing and return
        the array view of the resampled scan.
        :param ct_scan: Original ct_scan of shape (z, x, y).
        :param origin: Origin of the original ct_scan.
        :param spacing: Spacing of the original ct_scan.
        :param spacing_x: New spacing along dimension x.
        :param spacing_y: New spacing along dimension y.
        :param spacing_z: New spacing along dimension z.
        :return: An array (the resampled image) of shape (z, x, y).
        """
        img = LungsLoader._get_itk_from_scan(ct_scan, origin, spacing)
        # Compute new dimensions
        spacing = img.GetSpacing()
        size = img.GetSize()
        fact_x = spacing[0] / spacing_x
        fact_y = spacing[1] / spacing_y
        fact_z = spacing[2] / spacing_z
        size_x = int(round(size[0] * fact_x))
        size_y = int(round(size[1] * fact_y))
        size_z = int(round(size[2] * fact_z))
        # to do resampling
        f = sitk.ResampleImageFilter()
        f.SetReferenceImage(img)
        f.SetOutputOrigin(img.GetOrigin())
        f.SetOutputSpacing((spacing_x, spacing_y, spacing_z))
        f.SetSize((size_x, size_y, size_z))
        f.SetInterpolator(sitk.sitkLinear)
        result = f.Execute(img)
        return np.around(sitk.GetArrayFromImage(result))

    def get_scan(self, scan_id, resample=True):
        """
        Function to retrieve the data concerning a specific scan.
        :param scan_id: Scan id (the list is available through get_scan_ids).
        :param resample: Whether or not resampling need to be done. Boolean (defaults to True).
        :return: ct_scan of shape (z, x, y), origin, spacing
        """
        if scan_id not in self.scans:
            raise IndexError(
                f"Scan id {scan_id} not found in loaded scans. Make sure the file exists on this computer"
            )
        if self._cache:
            if scan_id not in self._cache_scans["not_resampled"]:
                (ct_scan, origin, spacing) = self._load_itk(self.scans[scan_id])
                self._cache_scans["not_resampled"][scan_id] = (ct_scan, origin, spacing)
                if resample:
                    resampled = self._resample(ct_scan, origin, spacing, 1, 1, 1)
                    self._cache_scans["resampled"][scan_id] = (
                        resampled, np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
                    )
                    return self._cache_scans["resampled"][scan_id]
                else:
                    return ct_scan, origin, spacing
        else:
            (ct_scan, origin, spacing) = self._load_itk(self.scans[scan_id])
            if resample:
                resampled = self._resample(ct_scan, origin, spacing, 1, 1, 1)
                return resampled, np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
            else:
                return ct_scan, origin, spacing

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
