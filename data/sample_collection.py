
# coding: utf-8

# ## Sample Collection
# 
# ### Author:
# Zachary Patel (zpatel@berkeley.edu)
# 
# ### Date:
# April 2020
# 
# ### Repository:
# https://github.com/zbpatel/DataHonorsThesis
# 
# 

# ### Description:
# This file contains the code used to collect performance profiles from a variety of basic testing networks generated with a given parameter set. 
# 
# TensorBoard output logs will be stored in tboard_logs/. (this can be changed by setting the value of log_base_dir)
# 
# After running this code, logs must be opened in TensorBoard and profiling data should be exported to CSV / JSON format for further processing.
# 
# 
# Information about the TensorFlow Profiler can be found: https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
# 

# ### Updating TensorFlow and TensorBoard
# 
# At the time of writing, the profiler is only included in the nightly releases of tensorflow
# (see the attached TensorFlow article for further reference)
# 
# Note: uninstalling and updating the following packages in the next line can take a while

# In[ ]:


#!pip uninstall -y -q tensorflow tensorboard
#!pip uninstall -y -q tensorflow tensorboard
#!pip install -U -q tf-nightly tb-nightly tensorboard_plugin_profile


# In[ ]:


#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from datetime import datetime
from packaging import version

import os

import tensorflow as tf
import numpy as np
print("TensorFlow version: ", tf.__version__)


# In[ ]:


device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# ### Configure Log Storage Filepath

# In[ ]:


log_base_dir = "tboard_logs/"


# ### Helper Functions

# In[ ]:


def construct_model(input_shape=(28, 28, 1), filters=1, kernel_size=[3, 3], strides=[1,1], padding="same", 
                    output_shape=1):
    """
    Constructs a simple network with specified parameters. 
    
    This network has 2 layers: 
        1. Conv2D with the specified parameters
        2. Dense to transform data to the proper number of class labels 
            (makes proper sizing of training data easier)
    
    Arguments:
        input shape: list of length 3
        filters: integer
        kernel_size: list of length 2
        strides: list of length 2
        padding: string
        classes: integer
        
    Returns:
        model: a keras model
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            input_shape = input_shape,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding
        ), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_shape, activation="sigmoid")
    ])

    return model

def compile_model(model):
    """
    Compiles the specified model with some fixed parameters
    
    Arguments:
        model: a keras model
        
    Returns:
        None
    """
    
    # compiling the model
    model.compile(
        loss = "binary_crossentropy",
        optimizer = "sgd",
        metrics = ["accuracy"]
    )


# In[ ]:


def configure_tboard_callback(log_loc):
    """
    Configures a TensorBoard callback with the profiler enabled and set to profile during the second batch

    Arguments:
        log_loc: string

    Returns:
        tboard_callback: a configured TensorBoard callback
    """
    # the most important setting here is profile batch (if this was left at 0, no profiling data would be recorded)
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_loc,
        histogram_freq = 0,
        update_freq = "epoch",
        profile_batch=2
    )
    
    return tboard_callback
    


# In[ ]:


def generate_dummy_data(samples, classes=2, h=28, w=28, c=1):
    """
    Given a variety of size parameters, generates a set of dummy data to be used for training a network
    
    Arguments:
        samples: number of random samples to generate
        classes: number of distinct "y" classes to label sample points
        h, w, c: dimensions of 
    
    Returns:
        dummy_data: (samples, h, w, c) np.array with values sampled from a uniform(0, 1) distribution
        dummy_labels: (samples, 1) np.array of integer values
    """
    
    dummy_data = np.random.random((samples, h, w, c))
    dummy_labels = np.random.randint(low=0, high=classes, size=(samples, 1))
    
    return dummy_data, dummy_labels


# ### Data Collection

# #### Configuring the range of parameters to test
# 
# For batch size, pick powers of 2 as well as 1 above and below each power of 2. Also pick 320, which is the number of TensorCores on the Turing T4 and might have interesting performance implications.
# 
# For stride length, filter count and filter size, pick from a set of reasonable and commonly used parameters

# In[ ]:


# configuring parameter ranges to test

# batch size ranges are all powers of 2 from 2 to 256 and one above / below each point
# also included are 320 +- 1 since there are 320 TensorCores on the Turing T4
powers_of_2 = np.power(2, np.arange(1, 8))
batch_size_range = np.concatenate((powers_of_2, powers_of_2 - 1, powers_of_2 + 1, np.array([319, 320, 321])))
# unique - removes duplicates, sort - makes the print statement nicer ;)
batch_size_range = np.sort(np.unique(batch_size_range))
print("Batch sizes: ", batch_size_range)

stride_range = range(1, 4)
print("Stride sizes: ", stride_range)

filter_count_range = range(1, 11)
print("Filter count range: ", filter_count_range)

kernel_size_range = range(1, 5, 2)
print("Kernel sizes: ", kernel_size_range)


# In[ ]:


print("Beginning Testing")
bs_count = 0
for batch_size in batch_size_range:
    bs_count += 1
    print("Testing batch size %d of %d" %(bs_count, batch_size_range.shape[0]))
    for stride_size in stride_range:
        for kernel_size in kernel_size_range:
            for filter_count in filter_count_range:
                # constructing and compiling the model to sample
                model = construct_model(
                    filters=filter_count, 
                    strides=[stride_size, stride_size], 
                    kernel_size=[kernel_size, kernel_size]
                )
                
                compile_model(model)

                # regenerating the callback for each run is required to give each sample a different name
                # this will produce a significant amount of warning text however
            
            
                # Label format:
                # timestamp-b(batch_size)s(stride_size)k(kernel_size)f(filter_count)
                log_loc =  log_base_dir + datetime.now().strftime("%Y%m%d-%H%M") + "-b%ds%dk%df%d" %(batch_size, stride_size, kernel_size, filter_count)
                tboard_callback = configure_tboard_callback(log_loc)

                
                # generating dummy data:
                # setting number of samples to batch_size * 2 since we set the profiler to sample on the second batch
                dummy_data, dummy_labels = generate_dummy_data(batch_size * 2)  
                
                # fit model for dummy data and collect logs (results)
                model.fit(
                    x=dummy_data, 
                    y=dummy_labels, 
                    batch_size = batch_size, 
                    callbacks = [tboard_callback]
                )

