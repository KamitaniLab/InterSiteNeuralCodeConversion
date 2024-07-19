import time
import datetime
import sys
from visdom import Visdom
import numpy as np
import torch
import glob
from fastl2lir import FastL2LiR
from bdpy.dataform import Features, load_array, save_array
import os, hdf5storage

class Logger():
    def __init__(self, n_iterations, batches_iteration):
        self.viz = Visdom()
        self.n_iterations = n_iterations
        self.batches_iteration = batches_iteration
        self.iteration = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\riteration %03d/%03d [%04d/%04d] -- ' % (self.iteration, self.n_iterations, self.batch, self.batches_iteration))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_iteration*(self.iteration - 1) + self.batch
        batches_left = self.batches_iteration*(self.n_iterations - self.iteration) + self.batches_iteration - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # End of iteration
        if (self.batch % self.batches_iteration) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.iteration]), Y=np.array([loss/self.batch]),opts={'xlabel': 'iterations', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.iteration]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next iteration
                self.losses[loss_name] = 0.0

            self.iteration += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

class LambdaLR():
    def __init__(self, n_iterations, offset, decay_start_iteration):
        assert ((n_iterations - decay_start_iteration) > 0), "Decay must start before the training session ends!"
        self.n_iterations = n_iterations
        self.offset = offset
        self.decay_start_iteration = decay_start_iteration

    def step(self, iteration):
        return 1.0 - max(0, iteration + self.offset - self.decay_start_iteration)/(self.n_iterations - self.decay_start_iteration)

def predict(X, W, b, y_mean, y_norm, ones):
    '''Predict with the fitted model.

    Parameters
    ----------
    X : tensor_like

    Returns
    -------
    Y : tensor_like
    '''
    # y_mean.ndim should be 2

    # Prediction
    y_pred_tmp = torch.matmul(X, W) + torch.matmul(ones, b)
    y_pred_tmp = y_pred_tmp * y_norm + y_mean

    return y_pred_tmp

def predict_normalized(X, W, b):
    '''Predict with the fitted linear model.

    Parameters
    ----------
    X : tensor_like

    Returns
    -------
    Y : tensor_like
    '''
    # y_mean.ndim should be 2

    # Reshape
    reshape_y = W.dim() > 2
    if reshape_y:
        Y_shape = W.shape
        W = W.reshape(W.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
    else:
        W = W
        b = b

    # Prediction
    Y = torch.matmul(X, W) + torch.matmul(torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device), b)

    if reshape_y:
        Y = Y.reshape((Y.shape[0],) + Y_shape[1:])

    return Y

def fastl2lir_parameter(model_store, chunk_axis=1):
    # W: shape = (n_voxels, shape_features)
    if os.path.isdir(os.path.join(model_store, 'W')):
        W_files = sorted(glob.glob(os.path.join(model_store, 'W', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'W.mat')):
        W_files = [os.path.join(model_store, 'W.mat')]
    else:
        raise RuntimeError('W not found.')

    # b: shape = (1, shape_features)
    if os.path.isdir(os.path.join(model_store, 'b')):
        b_files = sorted(glob.glob(os.path.join(model_store, 'b', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'b.mat')):
        b_files = [os.path.join(model_store, 'b.mat')]
    else:
        raise RuntimeError('b not found.')

    x_mean = hdf5storage.loadmat(os.path.join(model_store, 'x_mean.mat'))['x_mean']  # shape = (1, n_voxels)
    x_norm = hdf5storage.loadmat(os.path.join(model_store, 'x_norm.mat'))['x_norm']  # shape = (1, n_voxels)
    y_mean = hdf5storage.loadmat(os.path.join(model_store, 'y_mean.mat'))['y_mean']  # shape = (1, shape_features)
    y_norm = hdf5storage.loadmat(os.path.join(model_store, 'y_norm.mat'))['y_norm']  # shape = (1, shape_features)

    W = load_array(W_files[0], key='W')
    b = load_array(b_files[0], key='b')

    return x_mean, x_norm, y_mean, y_norm, W, b

def test_fastl2lir_div(model_store, x, i):
    # Load W
    if os.path.isdir(os.path.join(model_store, 'W')):
        W_files = sorted(glob.glob(os.path.join(model_store, 'W', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'W.mat')):
        W_files = [os.path.join(model_store, 'W.mat')]
    else:
        raise RuntimeError('W not found.')

    # Load b
    if os.path.isdir(os.path.join(model_store, 'b')):
        b_files = sorted(glob.glob(os.path.join(model_store, 'b', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'b.mat')):
        b_files = [os.path.join(model_store, 'b.mat')]
    else:
        raise RuntimeError('b not found.')

    # Check if i is valid
    if i >= len(W_files) or i >= len(b_files):
        raise ValueError(f'Invalid chunk index {i}. Number of chunks: {len(W_files)}.')

    device = x.device

    Wf = W_files[i]
    bf = b_files[i]

    W_tmp = torch.tensor(load_array(Wf, key='W'), dtype=x.dtype).to(device)
    b_tmp = torch.tensor(load_array(bf, key='b'), dtype=x.dtype).to(device)

    y_pred_tmp = predict_normalized(x, W_tmp, b_tmp)

    return y_pred_tmp


def compute_layer_weights(chunk_dnn_features, features_list):
    """
    Compute weights for each layer based on the norm of the features.

    Parameters:
    - dataloaders_batches: Dictionary containing batches of data for each layer.
    - features_list: List of layers for which weights are computed.
    - iteration: Current training iteration.

    Returns:
    - Dictionary of layer weights.
    """

    feat_norms = []

    # Compute norm for each layer's features
    for layer in features_list:
        # Extract the current layer's batch of features
        current_features = chunk_dnn_features[layer]

        # Compute the norm and append to the list
        feat_norms.append(np.linalg.norm(current_features))

    feat_norms = np.array(feat_norms, dtype='float32')

    # Compute weights using the inverse of squared norms
    weights = 1. / (feat_norms ** 2)
    # Normalize the weights
    weights /= weights.sum()

    # Map each layer to its corresponding weight
    layer_weights = dict(zip(features_list, weights))

    return layer_weights

class PathBuilder:
    def __init__(self, trg_dir, vgg_network, subject_trg, roi):
        self.trg_dir = trg_dir
        self.vgg_network = vgg_network
        self.subject_trg = subject_trg
        self.roi = roi

    def build_model_path(self, layer):
        """
        :param layer: layer name
        :return: the path
        """
        return os.path.join(self.trg_dir, self.vgg_network, layer, self.subject_trg, self.roi, 'model')

def dnn_chunk_get(chunk_dir, i):
    if os.path.isdir(os.path.join(chunk_dir, 'chunk')):
       dnn_chunks = sorted(glob.glob(os.path.join(chunk_dir, 'chunk', '*.mat')))
    elif os.path.isfile(os.path.join(chunk_dir, 'chunk.mat')):
        dnn_chunks = [os.path.join(chunk_dir, 'chunk.mat')]
    else:
        raise RuntimeError('W not found.')

    # Check if i is valid
    if i >= len(dnn_chunks) :
        raise ValueError(f'Invalid chunk index {i}. Number of chunks: {len(dnn_chunks)}.')
    # device = x.device
    dnn = dnn_chunks[i]
    dnn_tmp = hdf5storage.loadmat(dnn)['chunk']

    return dnn_tmp

def test_fastl2lir_revise(model_store, src_mean_std_store, x, chunk_axis=1):
    # W: shape = (n_voxels, shape_features)
    if os.path.isdir(os.path.join(model_store, 'W')):
        W_files = sorted(glob.glob(os.path.join(model_store, 'W', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'W.mat')):
        W_files = [os.path.join(model_store, 'W.mat')]
    else:
        raise RuntimeError('W not found.')

    # b: shape = (1, shape_features)
    if os.path.isdir(os.path.join(model_store, 'b')):
        b_files = sorted(glob.glob(os.path.join(model_store, 'b', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'b.mat')):
        b_files = [os.path.join(model_store, 'b.mat')]
    else:
        raise RuntimeError('b not found.')

    x_mean = hdf5storage.loadmat(os.path.join(model_store, 'x_mean.mat'))['x_mean']  # shape = (1, n_voxels)
    x_norm = hdf5storage.loadmat(os.path.join(model_store, 'x_norm.mat'))['x_norm']  # shape = (1, n_voxels)
    y_mean = hdf5storage.loadmat(os.path.join(src_mean_std_store, 'y_mean.mat'))['y_mean']  # shape = (1, shape_features)
    y_norm = hdf5storage.loadmat(os.path.join(src_mean_std_store, 'y_norm.mat'))['y_norm']  # shape = (1, shape_features)

    x = (x - x_mean) / x_norm

    # Prediction
    y_pred_list = []
    for i, (Wf, bf) in enumerate(zip(W_files, b_files)):
        print('Chunk %d' % i)

        start_time = time.time()
        W_tmp = load_array(Wf, key='W')
        b_tmp = load_array(bf, key='b')

        model = FastL2LiR(W=W_tmp, b=b_tmp)
        y_pred_tmp = model.predict(x)

        # Denormalize Y
        if y_mean.ndim == 2:
            y_pred_tmp = y_pred_tmp * y_norm + y_mean
        else:
            y_pred_tmp = y_pred_tmp * y_norm[:, [i], :] + y_mean[:, [i], :]

        y_pred_list.append(y_pred_tmp)

        print('Elapsed time: %f s' % (time.time() - start_time))

    return np.concatenate(y_pred_list, axis=chunk_axis)