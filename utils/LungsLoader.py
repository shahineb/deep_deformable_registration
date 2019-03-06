import os
import warnings
from collections import defaultdict
import numpy as np
import random
import imageio
import SimpleITK as sitk
from keras.preprocessing import image as image_prep

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
    def _resample(ct_scan, origin, spacing, spacing_x, spacing_y, spacing_z,
                  interpolator=sitk.sitkLinear):
        """
        Takes as input the informations about the scan to resample and the new spacing and return
        the array view of the resampled scan.
        :param ct_scan: Original ct_scan of shape (z, x, y).
        :param origin: Origin of the original ct_scan.
        :param spacing: Spacing of the original ct_scan.
        :param spacing_x: New spacing along dimension x.
        :param spacing_y: New spacing along dimension y.
        :param spacing_z: New spacing along dimension z.
        :param interpolator: Interpolator used for the resampling action (default to sitk.sitkLinear).
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
        f.SetInterpolator(interpolator)
        result = f.Execute(img)
        return np.around(sitk.GetArrayFromImage(result))

    @staticmethod
    def rescale_scan(ct_scan, origin, spacing, new_width, new_height, new_depth,
                     normalize=True, interpolator=sitk.sitkLinear):
        """
        Rescales a given scan using the SimpleITK resampler.
        :param ct_scan: Original data.
        :param origin: Original data.
        :param spacing: Original data.
        :param new_width: Width to be resized to (shape[1]).
        :param new_height: Height to be resized to (shape[2]).
        :param new_depth: Depth to be resized to (shape[0])
        :param normalize: Whether or not data should be normalized (default to True).
        :param interpolator: Interpolator used for the resampling action (default to sitk.sitkLinear).
        :return: The rescaled ct_scan, origin, spacing.
        """
        if not np.all(origin == 0):
            raise ValueError("Please feed only resampled images to this function.")
        if ct_scan.shape[2] != ct_scan.shape[1]:
            raise ValueError("Scans should have the same width and height.")
        img = LungsLoader._get_itk_from_scan(ct_scan, origin, spacing)
        f = sitk.ResampleImageFilter()
        f.SetReferenceImage(img)
        f.SetOutputOrigin(img.GetOrigin())
        spacing_x, spacing_y, spacing_z = img.GetSpacing()
        size_x, size_y, size_z = img.GetSize()
        spacing_x /= (new_width / size_x)
        spacing_y /= (new_height / size_y)
        spacing_z /= (new_depth / size_z)
        size_x = new_width
        size_y = new_height
        size_z = new_depth
        f.SetOutputSpacing((spacing_x, spacing_y, spacing_z))
        f.SetSize((size_x, size_y, size_z))
        f.SetInterpolator(interpolator)
        result = f.Execute(img)
        new_scan = sitk.GetArrayFromImage(result)
        if normalize:
            new_scan = (new_scan - np.mean(new_scan)) / np.var(new_scan)
        return new_scan, result.GetOrigin(), result.GetSpacing()

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

    @staticmethod
    def clip_scan(ct_scan):
        _, x, _ = ct_scan.shape
        clipping_min = np.min(ct_scan[:, x // 2, 2:5])
        clipping_min = min(clipping_min, np.min(ct_scan[:, x // 2, -5:-2]))
        return np.clip(ct_scan, clipping_min, None)

    def preprocess_scans(self, scan_ids, width, height, depth, clipping=True, loop=False,
                         shuffle=False):
        """
        Preprocess a bulk of scan by rescaling each image to the target dimensions.
        :param scan_ids: List of the scan ids to preprocess.
        :param width: Target width for the preprocessed scans.
        :param height: Target height for the preprocessed scans.
        :param depth: Target depth for the preprocessed scans.
        :param clipping: Whether the scans should be clipped
        :param loop: If true, endlessly loop on data (default: false).
        :param shuffle: If true, scans are shuffled (default: false)
        :return: A generator of ct_scan, origin, spacing tuples.
        """
        while True:
            if shuffle:
                random.shuffle(scan_ids)
            for scan_id in scan_ids:
                ct_scan, origin, spacing = self.get_scan(scan_id, resample=True)
                if clipping:
                    scan, origin, spacing = self.rescale_scan(
                        ct_scan, origin, spacing, width, height, depth, normalize=True
                        )
                    yield self.clip_scan(scan), origin, spacing
                else:
                    yield self.rescale_scan(
                        ct_scan, origin, spacing, width, height, depth, normalize=True
                    )
            if not loop:
                break

    def toy_preprocess_scans(self, scan_ids, width, height, depth, clipping=True, loop=False,
                             seed=42, shuffle=False):
        """
        Mimic the preprocess_scans function but yields slightly transformed target images.
        The source is yielded first and then the target (pair index are src and impair index are
        targets if it was a list).
        :param scan_ids: See preprocess scans.
        :param width: See preprocess scans.
        :param height: See preprocess scans.
        :param depth: See preprocess scans.
        :param clipping: See preprocess scans.
        :param loop: See preprocess scans.
        :param shuffle: See preprocess scans.
        :param seed: Seed used to initialize numpy random and make the dataset reproducible.
        Defaults to 42.
        :return: A generator of ct_scan, origin, spacing tuples. Alternates between a source scan
        and a target created with small transformations.
        """
        # Initialize image transformer
        kwds_generator = {'rotation_range': 5,
                          'width_shift_range': 0.03,
                          'height_shift_range': 0.03,
                          'zoom_range': 0.03,
                          'data_format': "channels_first",  # z axis is first
                          }
        image_gen = image_prep.ImageDataGenerator(**kwds_generator)

        scan_gen = self.preprocess_scans(scan_ids, width, height, depth, clipping, loop, shuffle)
        for ct_scan, origin, spacing in scan_gen:
            yield ct_scan, origin, spacing
            transformed_scan = image_gen.random_transform(ct_scan, seed=seed)
            if seed is not None:
                seed += 1
            yield transformed_scan, origin, spacing

    def preprocess_segmentations(self, scan_ids, width, height, depth, ohe=None, loop=False, shuffle=False):
        """
        Retrieves the segmentation data for a bulk of scans and rescales it to the target dimensions.
        :param scan_ids: List of the scan ids to retrieve.
        :param width: Target width for the retrieved segmentation arrays.
        :param height: Target height for the retrieved segmentation arrays.
        :param depth: Target depth for the retrieved segmentation arrays.
        :param ohe: SKlearn one hot encoder trained to process the data. If this argument is none,
        no encoding is performed. If an encoder is given the data is preprocessed to encode the
        classes.
        :param loop: If true, endlessly loop on data (default: false).
        :param shuffle: If true, scans are shuffled (default: false)
        :return: A generator of (scan array, origin, spacing).
        """
        seg_folder = os.path.join(self._data_folder, "seg-lungs-LUNA16")
        while True:
            if shuffle:
                random.shuffle(scan_ids)
            for scan_id in scan_ids:
                f = os.path.join(seg_folder, scan_id + ".mhd")
                ct_scan, origin, spacing = self._load_itk(f)
                ct_scan = self._resample(
                    ct_scan, origin, spacing, 1, 1, 1, interpolator=sitk.sitkNearestNeighbor
                    )
                ct_scan, origin, spacing = self.rescale_scan(
                    ct_scan, np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]),
                    width, height, depth, normalize=False, interpolator=sitk.sitkNearestNeighbor
                    )
                if ohe is None:
                    yield ct_scan, origin, spacing
                else:
                    shape = ct_scan.shape
                    n_classes = len(ohe.categories_[0])
                    ct_scan = ohe.transform(
                        ct_scan.flatten().reshape(-1, 1)
                        ).toarray().reshape(shape + (n_classes,))
                    yield ct_scan, origin, spacing
            if not loop:
                break

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

    @staticmethod
    def _create_gif(images, target, time_per_image=0.1):
        images_raw = []
        for f in images:
            images_raw.append(imageio.imread(f))
        imageio.mimsave(target, images_raw, duration=len(images) * time_per_image)

    @staticmethod
    def create_gifs(folder):
        """
        Create gifs from the images recorder in ./folder/observations_xx/
        :param folder:
        :return:
        """
        # Retrieve images paths
        images_dict = defaultdict(list)
        folders_sorting_key = lambda s: int(s.split("_")[-1])
        obs_folders = sorted(os.listdir(folder), key=folders_sorting_key)
        for obs_folder in obs_folders:
            for f in os.listdir(os.path.join(folder, obs_folder)):
                image_name = "_".join(f.split("_")[:-1])
                images_dict[image_name].append(os.path.join(folder, obs_folder, f))
        # Create gifs
        for name in images_dict:
            target = os.path.join(folder, name + ".gif")
            LungsLoader._create_gif(images_dict[name], target)
