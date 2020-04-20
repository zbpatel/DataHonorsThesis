# Zachary Patel
# Univ. California, Berkeley
# zpatel@berkeley.edu

from datetime import datetime
import tensorflow_gpu as tf
import os

# defining a model that is just a single convolution layer with the given params
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        input_shape = (10, 10, 1),
        filters = 10,
        kernel_size = [3, 3],
        strides = [1, 1],
        padding = "same"
    )
])

# compiling the model
model.compile()

# Registering callback to record logs for TensorBoard
log_loc = "tboard_logs/" +
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_loc,
    histogram_freq = 0,
    update_freq = "epoch")

print("Started Training\n")

# training the model
model.fit(data_set,
    epochs = 1,
    callbacks = [tboard_callback]
    )

print("Training Completed\n")
