# -*- coding: utf-8 -*-

import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

from constants import PLOT_DIR


def plot_images(data_handler):
    batches_test = data_handler.get_train_batch_streamer(16)
    batch = next(batches_test)
    
    n = 10
    plt.figure(figsize=(n, 2))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(batch["X"][i,0,:,:])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()

# =============================================================================
# PCA plot of latent space
# =============================================================================

def plot_PCA(encoder, labeled_data_handler):
    batches_test = labeled_data_handler.get_test_batch_streamer(256)
    batch = next(batches_test)
    X = batch['X']
    y = batch['y']
    latent = encoder.predict(X)
    
    latent_flattened = latent.reshape((latent.shape[0], -1))
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(latent_flattened)
    
    plt.scatter(projected[:, 0], projected[:, 1],
                c=y, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('spectral', 50))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()

# =============================================================================
# Plot 16 decoded test images
# =============================================================================

def plot_prediction(ae, data_handler, save_name=None):
#    total = 50000
#    val_size = total // 5
#    data_handler = DataHandler(bands=28, test_size=16, 
#                                val_size=val_size, 
#    #                           val_size=32, train_size=32,
#                               dataset="ESC-US", hop_length=6*1024, window=12*1024,
#                               # label_whitelist=["101 - Dog", "201 - Rain"]
#                               )
    batches_test = data_handler.get_test_batch_streamer(16)
    batch = next(batches_test)
    decoded_imgs = ae.predict(batch["X"])
    
    n = decoded_imgs.shape[0]
    plt.figure(figsize=(n, 2))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(batch["X"][i,0,:,:])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_imgs[i,0,:,:])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    if save_name:
        plt.savefig(os.path.join(PLOT_DIR, save_name + ".png"))
    else:
        plt.show()

# Adapted from https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """

    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

def plot_conv_weights(model, layer_name):
    # Visualize weights
    layer = model.layers[layer_name] if isinstance(layer_name, int) \
            else model.get_layer(layer_name)
    W = layer.kernel.get_value(borrow=True)
    W = np.squeeze(W)
    
    # we use channel_first. channel_last is expected
    W = np.moveaxis(W, 0, -1) 
    W = np.moveaxis(W, 0, -1)

    # TODO: only diplay a subset

    if len(W.shape) == 4:
        W = W.reshape((-1,W.shape[2],W.shape[3]))
    print("W shape : ", W.shape)

    pl.figure(figsize=(10, 10))
    pl.title('conv weights')
    s = int(np.sqrt(W.shape[0])+1)
    nice_imshow(pl.gca(), make_mosaic(W, s, s), cmap=cm.binary)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

