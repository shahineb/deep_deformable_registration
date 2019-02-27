import keras.layers as kl
from keras.models import Model


class UNet3D:
    def __init__(self, input_shape, n_classes, depth, n_base_filters=64):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.depth = depth
        self.n_base_filters = n_base_filters

    def build(self):
        # Initialisation
        input_layer = kl.Input(shape=self.input_shape, name="input_layer")
        cur_layer = input_layer
        levels = []

        # Downward
        for i in range(self.depth):
            n_kernels = self.n_base_filters * (2 ** i)
            conv_a = kl.Conv3D(
                n_kernels, kernel_size=3, activation="relu", padding="same", name=f"conv_a_{i}"
            )(cur_layer)
            conv_b = kl.Conv3D(
                n_kernels, kernel_size=3, activation="relu", padding="same", name=f"conv_b_{i}"
            )(conv_a)

            if i < self.depth - 1:
                pool = kl.MaxPool3D(
                    pool_size=(2, 2, 2), padding="same", name=f"pool_{i}"
                )(conv_b)
                cur_layer = pool
                levels.append([conv_a, conv_b, pool])
            else:
                cur_layer = conv_b
                levels.append([conv_a, conv_b])

        # Upward
        for i in range(self.depth - 2, -1, -1):
            n_kernels = self.n_base_filters * (2 ** i)
            up = kl.UpSampling3D(size=(2, 2, 2), name=f"upsampling_{i}")(cur_layer)
            upconv_a = kl.Conv3D(
                n_kernels, kernel_size=3, activation="relu", padding="same", name=f"upconv_a_{i}"
            )(up)
            merge = kl.Concatenate(axis=4, name=f"merge{i}")([levels[i][1], upconv_a])
            upconv_b = kl.Conv3D(
                n_kernels, kernel_size=3, activation="relu", padding="same", name=f"upconv_b_{i}"
            )(merge)
            upconv_c = kl.Conv3D(
                n_kernels, kernel_size=3, activation="relu", padding="same", name=f"upconv_c_{i}"
            )(upconv_b)
            cur_layer = upconv_c

        # Finalisation
        output_layer = kl.Conv3D(
            self.n_classes, 1, activation="sigmoid", name="output_layer"
        )(cur_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model
