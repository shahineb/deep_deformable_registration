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
        """
        Plot a display of a ct scan with a cursor to explore the layers.
        :param ct_scan: Ct scan to display
        :return: None. Plot the scan.
        """
        int_slider = widgets.IntSlider(
            min=0, max=ct_scan.shape[0] - 1, step=1, value=10
            )
        interact(self.show, z=int_slider, ct_scan=fixed(ct_scan))

    def display_n_slices(self, ct_scan, n, axes=None, return_fig=False):
        prev_figsize = self._plt.figure().get_size_inches()
        cols = 4
        rows = max(n // 4, 1)
        step = ct_scan.shape[0] // n
        fig, ax = self._plt.subplots(rows, cols, figsize=(16, rows * 4))
        for i in range(n):
            cur_slice = ct_scan[step * i, :, :]
            if rows > 1:
                ax[i // cols][i % cols].imshow(cur_slice, cmap=self._plt.cm.gray)
                ax[i // cols][i % cols].set_title(f"Slice {step * i}")
            else:
                ax[i % cols].imshow(cur_slice, cmap=self._plt.cm.gray)
                ax[i % cols].set_title(f"Slice {step * i}")
        self._plt.tight_layout()
        self._plt.figure(figsize=prev_figsize)
        self._plt.show()
        if return_fig:
            return fig, ax

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
        
    def plot_def(def_field, axis, height, return_fig=False):
        '''axis = 0 : z= height
            axis = 1 : x= height
            axis = 2 : y = height
            def field of shape (1, z, x, y, 3)

        '''
        fig = self._plt.figure()
        ax = fig.add_subplot(111)
        if axis == 0:
            # fixed z
            assert (height >= def_field[0].shape[axis]),"Height is bigger than shape"
            x = np.linspace(-2, 2, def_field[0,height,:,:,1].shape[0])
            y = np.linspace(-2, 2, def_field[0,height,:,:,1].shape[1])
            def_f_x = def_field[0,height,:,:,1]
            def_f_y = def_field[0,height,:,:,2]
            color = 2 * np.log(np.hypot(def_f_x, def_f_y))
            ax.streamplot(x, y, def_f_x, def_f_y, color=color, linewidth=1, cmap=self._plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            ax.set_aspect('equal')
            self._plt.show()
            if return_fig:
                return fig, ax
        elif axis == 1:
            # fixed x
            assert (height >= def_field[0].shape[axis]),"Height is bigger than shape"
            x = np.linspace(-2, 2, def_field[0,:,height,:,0].shape[0])
            y = np.linspace(-2, 2, def_field[0,:,height,:,0].shape[1])
            def_f_x = def_field[0,:,height,:,0]
            def_f_y = def_field[0,:,height,:,2]
            color = 2 * np.log(np.hypot(def_f_x, def_f_y))
            ax.streamplot(x, y, def_f_x, def_f_y, color=color, linewidth=1, cmap=self._plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            ax.set_aspect('equal')
            self._plt.show()
            if return_fig:
                return fig, ax
        elif axis == 2:
                # fixed y
            assert (height >= def_field[0].shape[axis]),"Height is bigger than shape"
            x = np.linspace(-2, 2, def_field[0,:,:,height,0].shape[0])
            y = np.linspace(-2, 2, def_field[0,:,:,height,0].shape[1])
            def_f_x = def_field[0,:,:,height,0]
            def_f_y = def_field[0,:,:,height,1]
            color = 2 * np.log(np.hypot(def_f_x, def_f_y))
            ax.streamplot(x, y, def_f_x, def_f_y, color=color, linewidth=1, cmap=self._plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            ax.set_aspect('equal')
            self._plt.show()
            if return_fig:
                return fig, ax 
            
    def return_def(self, def_field, axis, height):
        '''axis = 0 : z= height
                axis = 1 : x= height
                axis = 2 : y = height
                def field of shape (1, z, x, y, 3)

            '''
        if axis == 0:
            # fixed z
            assert (height <= def_field[0].shape[axis]),"Height is bigger than shape"
            x = np.linspace(-2, 2, def_field[0,height,:,:,1].shape[0])
            y = np.linspace(-2, 2, def_field[0,height,:,:,1].shape[1])
            def_f_x = def_field[0,height,:,:,1]
            def_f_y = def_field[0,height,:,:,2]
            label_x = '$x$'
            label_y = '$y$'
            return [x, y, def_f_x, def_f_y, label_x, label_y]
        elif axis == 1:
            # fixed x
            assert (height <= def_field[0].shape[axis]),"Height is bigger than shape"
            x = np.linspace(-2, 2, def_field[0,:,height,:,0].shape[0])
            y = np.linspace(-2, 2, def_field[0,:,height,:,0].shape[1])
            def_f_x = def_field[0,:,height,:,0]
            def_f_y = def_field[0,:,height,:,2]
            label_x = '$z$'
            label_y = '$y$'
            return [x, y, def_f_x, def_f_y, label_x, label_y]
        elif axis == 2:
            # fixed y
            assert (height <= def_field[0].shape[axis]),"Height is bigger than shape"
            x = np.linspace(-2, 2, def_field[0,:,:,height,0].shape[0])
            y = np.linspace(-2, 2, def_field[0,:,:,height,0].shape[1])
            def_f_x = def_field[0,:,:,height,0]
            def_f_y = def_field[0,:,:,height,1]
            label_x = '$z$'
            label_y = '$x$'
            return [x, y, def_f_x, def_f_y, label_x, label_y]      

    def display_n_def(self, def_field, n, axis, return_fig=False):
        """def_field, field of shape (1, z, x, y, 3)
        n : number of slices
        axis : axis along
        
        """
        prev_figsize = self._plt.figure().get_size_inches()
        cols = 4
        rows = max(n // 4, 1)
        step = def_field[0].shape[axis] // n
        fig, ax = self._plt.subplots(rows, cols, figsize=(16, rows * 4))
        for i in range(n):
            if rows > 1:
                [x, y, def_f_x, def_f_y, label_x, label_y] = self.return_def(def_field, axis, step*i)
                color = 2 * np.log(np.hypot(def_f_x, def_f_y))
                ax[i // cols][i % cols].streamplot(x, y, def_f_x, def_f_y, color=color, linewidth=1, cmap=self._plt.cm.inferno,
                                                   density=2, arrowstyle='->', arrowsize=1.5)
                ax[i // cols][i % cols].set_xlabel(label_x)
                ax[i // cols][i % cols].set_ylabel(label_y)
                ax[i // cols][i % cols].set_xlim(-2,2)
                ax[i // cols][i % cols].set_xlim(-2,2)
                ax[i // cols][i % cols].set_aspect('equal')
                ax[i // cols][i % cols].set_title(f"Slice {step * i}")
            else:
                [x, y, def_f_x, def_f_y, label_x, label_y] = self.return_def(def_field, axis, step*i)
                color = 2 * np.log(np.hypot(def_f_x, def_f_y))
                ax[i % cols].streamplot(x, y, def_f_x, def_f_y, color=color, linewidth=1, cmap=self._plt.cm.inferno,
                                                   density=2, arrowstyle='->', arrowsize=1.5)
                ax[i % cols].set_xlabel(label_x)
                ax[i % cols].set_ylabel(label_y)
                ax[i % cols].set_xlim(-2,2)
                ax[i % cols].set_xlim(-2,2)
                ax[i % cols].set_aspect('equal')
                ax[i % cols].set_title(f"Slice {step * i}")
        self._plt.tight_layout()
        self._plt.figure(figsize=prev_figsize)
        self._plt.show()
        if return_fig:
            return fig, ax
        
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
    