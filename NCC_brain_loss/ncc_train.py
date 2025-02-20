
'''
Script for training the neural code converters using brain loss.
'''

import os
import warnings
import itertools
from time import time

import numpy as np
import scipy.io as sio
import pandas as pd

import bdpy
from bdpy.distcomp import DistComp
from bdpy.util import makedir_ifnot
from bdpy.dataform import save_array

from fastl2lir import FastL2LiR


# Main #######################################################################

def main():
    # Data settings ----------------------------------------------------
    # Brain data
    converter_param = './params/converter_params_VC.csv'
    df_param = pd.read_csv(converter_param)
    brain_dir = '../data/fmri'
    subjects_list = {'sub01': 'sub-01_NaturalImageTraining.h5',
                     'sub02': 'sub-02_NaturalImageTraining.h5',
                     'sub03': 'sub-03_NaturalImageTraining.h5',
                     'sub04': 'sub-04_NaturalImageTraining.h5',
                     'sub05': 'sub-05_NaturalImageTraining.h5'
                     }

    data_brain = {subject: bdpy.BData(os.path.join(brain_dir, dat_file))
                  for subject, dat_file in subjects_list.items()}

    label_name = 'image_index'

    rois_list = {'VC': 'ROI_VC =1'}

    # Results directory
    results_dir_root = './NCconverter_results'
    tmp_dir = './tmp'

    analysis_basename = os.path.splitext(os.path.basename(__file__))[0]

    for index, row in df_param.iterrows():
        src = str(row['Source'])
        trg = str(row['Target'])
        roi = str(row['ROI'])
        method = str(row['Method'])
        alpha = int(row['Alpha'])
        num_samples = int(row['Number of samples'])
        print('--------------------')
        print('Source: %s' % src)
        print('Target: %s' % trg)
        print('ROI: %s' % roi)
        print('alpha: %s' % alpha)
        print('Number of samples: %s' % num_samples)
        print('Method: %s' % method)
        # Setup
        # -----
        conversion = src + '_2_' + trg
        analysis_id = analysis_basename + '-' + conversion + '-' + roi + '-' + method + '-' + str(
            num_samples) + '-' + str(alpha)
        results_dir = os.path.join(results_dir_root, analysis_basename, conversion, roi, method,
                                   str(num_samples), 'model')
        makedir_ifnot(results_dir)
        makedir_ifnot(tmp_dir)

        # Check whether the analysis has been done or not.
        result_model = os.path.join(results_dir, 'NCconverter.mat')
        if os.path.exists(result_model):
            print('%s already exists and skipped' % result_model)
            continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()


        # Brain data
        x = data_brain[src].select(rois_list[roi])  # Brain data
        x_labels = data_brain[src].select(label_name)  # Image labels in the brain data

        y = data_brain[trg].select(rois_list[roi])
        y_labels = data_brain[trg].select(label_name)

        print('Total elapsed time (data preparation): %f' % (time() - start_time))

        # Model training
        # --------------
        print('Model training')
        start_time = time()
        train_NCconverter(x, y,
                          x_labels, y_labels,
                          # embedded_idxs_src, embedded_idxs_trg,
                          num_samples,
                          alpha=alpha,
                          output=results_dir, save_chunk=True,
                          axis_chunk=1, tmp_dir='tmp',
                          comp_id=analysis_id)

        print('Total elapsed time (model training): %f' % (time() - start_time))

    print('%s finished.' % analysis_basename)


# Functions ##################################################################
def train_NCconverter(x, y, x_labels, y_labels,
                      num_sample,
                      alpha=1.0,
                      output='./NCconverter_results.mat', save_chunk=False,
                      axis_chunk=1, tmp_dir='./tmp',
                      comp_id=None):
    if y.ndim == 4:
        # The Y input to the NCconveter has to be strictly number of samples x number of features
        y = y.reshape((y.shape[0], -1))
    elif y.ndim == 2:
        pass
    else:
        raise ValueError('Unsupported feature array shape')

    # Sort samples ------------------------------------------------------

    x_index = np.argsort(x_labels.flatten())
    x_labels = x_labels[x_index]
    x = x[x_index, :]

    y_index = np.argsort(y_labels.flatten())
    y_labels = y_labels[y_index]
    y = y[y_index, :]

    # Preprocessing ----------------------------------------------------------
    print('Preprocessing')
    start_time = time()

    # Normalize X (source fMRI data)
    x_mean = np.mean(x, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
    x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]
    x_normalized = (x - x_mean) / x_norm

    # Normalize Y (target fMRI data)
    y_mean = np.mean(y, axis=0)[np.newaxis, :]
    y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]
    y_normalized = (y - y_mean) / y_norm

    print('Elapsed time: %f' % (time() - start_time))

    # Model training loop ----------------------------------------------------
    start_time = time()
    print('Training')
    # Model training
    # conversion matrix for whole VC.

    model = FastL2LiR()
    # All voxels are used in training.
    model.fit(x_normalized, y_normalized, alpha=alpha, n_feat=0,
              chunk_size=0, dtype=np.float32)
    W = model.W
    b = model.b.flatten()  # b in fact has no effect as the data has been normalized, but it is kept for consistency.
    b= b.reshape(1,-1)
    M = np.vstack([W, b])

    # Save chunk results
    result_model = os.path.join(output, 'NCconverter.mat')
    save_array(result_model, M, 'M', dtype=np.float32, sparse=False)
    print('Saved %s' % result_model)

    etime = time() - start_time
    print('Elapsed time: %f' % etime)

    # Save results -----------------------------------------------------------
    print('Saving normalization parameters.')
    norm_param = {'x_mean': x_mean, 'y_mean': y_mean,
                  'x_norm': x_norm, 'y_norm': y_norm}
    save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
    for sv in save_targets:
        save_file = os.path.join(output, sv + '.mat')
        if not os.path.exists(save_file):
            try:
                save_array(save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                print('Saved %s' % save_file)
            except IOError:
                warnings.warn('Failed to save %s. Possibly double running.' % save_file)

    if not save_chunk:
        # Merge results into 'model'mat'
        raise NotImplementedError('Result merging is not implemented yet.')

    return None


def compute_roi_index(data, embeded_roi, roi_mapping):
    '''
    To aggregate the conversion matrices for each brain area,
    the embedded indices for each brain area in the VC is necessary.
    '''
    _, base_idx = data.select(embeded_roi, return_index=True)
    del _

    idx_loc = np.where(base_idx)[0]
    idx_mapper = dict(zip(idx_loc, range(len(idx_loc))))

    idxs = {}
    for roi in roi_mapping:
        _, idx = data.select(roi, return_index=True)

        loc = np.where(idx)[0]
        idxs[roi] = [idx_mapper[l] for l in loc]

    return idxs


# Entry point ################################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()