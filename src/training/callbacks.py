import keras


class LossHistory(keras.callbacks.Callback):
    # TODO : find a way to define loss history recording through callbacks
    # Right now, problem loss format is {'loss': [loss_by_epoch]} for chosen compilation loss
    # and {'output_layer_name': [loss_by_epoch]} for additional losses
    # Need to find how to retrieve this information from logs

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def plot_loss(self):
        pass
