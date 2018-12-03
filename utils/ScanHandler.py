import os
from ipywidgets import interact, fixed, widgets
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Change cur dir to project root
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


class ScanHandler:

    def __init__(self, plt):
        """
        Initializes the scan handler to manipulate CT scans.
        :param plt: pyplot object used for display
        """
        self._plt = plt

    def show(self, ct_scan, z):
        """
        Displays one layer of a ct scan.
        :param ct_scan: ct_scan of shape (z, x, y)
        :param z: layer to display
        :return: None
        """
        self._plt.imshow(ct_scan[z, :, :], cmap=self._plt.cm.gray)

    def interact_display(self, ct_scan):
        int_slider = widgets.IntSlider(
            min=0, max=ct_scan.shape[0] - 1, step=1, value=10
            )
        interact(self.show, z=int_slider, ct_scan=fixed(ct_scan))

    def display_n_slices(self, ct_scan, n):
        prev_figsize = self._plt.figure().get_size_inches()
        cols = 4
        rows = n // 4 + 1
        step = ct_scan.shape[0] // n
        self._plt.figure(figsize=(16, rows * 4))
        for i in range(n):
            self._plt.subplot(rows, cols, i + 1)
            cur_slice = ct_scan[step * i, :, :]
            self._plt.imshow(cur_slice, cmap=self._plt.cm.gray)
            self._plt.title(f"Slice {step * i}")
        self._plt.tight_layout()
        self._plt.show()
        self._plt.figure(figsize=prev_figsize)

    def show_vol(self, ct_scan, angle=0, save_path=None):
        """
        Displays volume view of ct_scan
        :param ct_scan: ct_scan of shape (z, x, y)
        :param angle: volume orientation wrt z axis (degrees)
        :return: None
        """
        def explode(data):
            shape_arr = np.array(data.shape)
            size = shape_arr[:3] * 2 - 1
            exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
            exploded[::2, ::2, ::2] = data
            return exploded

        def expand_coordinates(indices):
            x, y, z = indices
            x[1::2, :, :] += 1
            y[:, 1::2, :] += 1
            z[:, :, 1::2] += 1
            return x, y, z

        def normalize(cube):
            max_val = np.max(cube)
            min_val = np.min(cube)
            cube = (cube - min_val) / (max_val - min_val)
            return cube
        vol_size = max(ct_scan.shape)
        ct_scan = normalize(ct_scan)

        facecolors = self._plt.cm.viridis(ct_scan)
        facecolors[:, :, :, -1] = ct_scan
        facecolors = explode(facecolors)

        filled = facecolors[:, :, :, -1] != 0
        x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

        fig = self._plt.figure(figsize=(20 / 2.54, 18 / 2.54))
        ax = fig.gca(projection='3d')
        ax.view_init(30, angle)
        ax.set_xlim(right=vol_size * 2)
        ax.set_ylim(top=vol_size * 2)
        ax.set_zlim(top=vol_size * 2)

        ax.voxels(x, y, z, filled, facecolors=facecolors)
        self._plt.tight_layout()
        if save_path:
            self._plt.savefig(save_path)
        self._plt.show()

    @staticmethod
    def reduce(ct_scan, ratio=0.5, strategy="skipping"):
        """
        Reduce the dimensions of a ct_scan.
        :param ct_scan: scan to reduce.
        :param ratio: transformation ratio (default to 0.5, image size is divided by 2).
        :param strategy: strategy for the reduction (default to skipping).
        :return: reduced ct_scan.
        """
        if strategy == "skipping":
            return ScanHandler._reduce_skipping(ct_scan, ratio)
        elif strategy == "averaging":
            return ScanHandler._reduce_averaging(ct_scan, ratio)
        else:
            return NotImplementedError(f"CT Scan reduction strategy {strategy} doesn't exist.")

    @staticmethod
    def _reduce_skipping(ct_scan, ratio):
        step = int(1 / ratio)
        return ct_scan[:, ::step, ::step]

    @staticmethod
    def _reduce_averaging(ct_scan, ratio):
        raise NotImplementedError("CT scan reduction straty averaging is not implemented yet.")
