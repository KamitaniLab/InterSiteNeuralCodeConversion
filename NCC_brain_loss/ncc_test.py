'''Feature prediction with neural code conversion: prediction (test) script'''
import os
import glob
import numpy as np
import pandas as pd
import hdf5storage
import bdpy
from time import time
from bdpy.util import makedir_ifnot
from bdpy.dataform import load_array, save_array
from fastl2lir import FastL2LiR

# Main #######################################################################

def main():
    # Read settings
    converter_param = './params/converter_params_VC.csv'
    df_param = pd.read_csv(converter_param)
    # Brain data
    brain_dir = '../data/fmri'
    subjects_list = {'sub01': 'sub-01_NaturalImageTest.h5',
                     'sub02': 'sub-02_NaturalImageTest.h5',
                     'sub03': 'sub-03_NaturalImageTest.h5',
                     'sub04': 'sub-04_NaturalImageTest.h5',
                     'sub05': 'sub-05_NaturalImageTest.h5'
                     }

    label_name = 'image_index'
    rois_list = {'VC': 'ROI_VC =1'}

    # Converter models
    nc_models_dir_root = os.path.join('./NCconverter_results', 'ncc_train')

    # Load data
    print('Loading data')
    data_brain = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file)) for sbj, dat_file in subjects_list.items()}

    # Initialize directories
    makedir_ifnot('tmp')

    # Analysis loop
    print('Starting analysis loop')

    for _, row in df_param.iterrows():
        src, trg, roi = row['Source'], row['Target'], row['ROI']
        method, alpha, num_samples = row['Method'], int(row['Alpha']), int(row['Number of samples'])

        print(f'Source: {src}, Target: {trg}, ROI: {roi}, Alpha: {alpha}, Number of samples: {num_samples}, Method: {method}')

        # Brain data
        x = data_brain[src].select(rois_list[roi])
        x_labels = data_brain[src].select(label_name)
        y = data_brain[trg].select(rois_list[roi])
        conversion = f'{src}_2_{trg}'

        # Prepare data
        print('Preparing data')
        start_time = time()

        x_test_labels_unique = np.unique(x_labels)
        x_test_averaged = np.vstack([np.mean(x[(x_labels == lb).flatten(), :], axis=0) for lb in x_test_labels_unique])

        print(f'Total elapsed time (data preparation): {time() - start_time:.2f} s')

        # Convert x_test_averaged
        nc_models_dir = os.path.join(nc_models_dir_root, conversion, roi, method, str(num_samples), 'model')
        x_test_averaged_pred = test_ncconverter(nc_models_dir, x_test_averaged)

        # Prediction
        network = 'caffe/VGG_ILSVRC_19_layers'
        decoders_dir = '../data/pre-trained/decoders/ImageNetTraining/deeprecon_pyfastl2lir_alpha100_vgg19_allunits'
        features_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                         'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                         'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                         'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                         'fc6', 'fc7', 'fc8'][::-1]

        for feat in features_list:
            print(f'Predicting for layer: {feat}')
            start_time = time()
            pred_dnn = test_fastl2lir_div(os.path.join(decoders_dir, network, feat, trg, roi, 'model'), x_test_averaged_pred)
            print(f'Total elapsed time (prediction): {time() - start_time:.2f} s')

            # Save predicted features
            results_dir_prediction = os.path.join('./result_vgg', conversion, network, feat, 'target', roi)
            makedir_ifnot(results_dir_prediction)

            for i, label in enumerate(x_test_labels_unique):
                save_file = os.path.join(results_dir_prediction, f'{label}.mat')
                save_array(save_file, np.array([pred_dnn[i]]), 'feat', dtype=np.float32)
            print(f'Saved {results_dir_prediction}')

# Functions ##################################################################
def test_ncconverter(model_store, x):
    # Load NC converter
    print('Load NC converter')
    NCconverter = hdf5storage.loadmat(os.path.join(model_store, 'NCconverter.mat'))
    M = NCconverter['M']

    x_mean = hdf5storage.loadmat(os.path.join(model_store, 'x_mean.mat'))['x_mean']  # shape = (1, n_voxels)
    x_norm = hdf5storage.loadmat(os.path.join(model_store, 'x_norm.mat'))['x_norm']  # shape = (1, n_voxels)
    y_mean = hdf5storage.loadmat(os.path.join(model_store, 'y_mean.mat'))['y_mean']  # shape = (1, shape_features)
    y_norm = hdf5storage.loadmat(os.path.join(model_store, 'y_norm.mat'))['y_norm']  # shape = (1, shape_features)

    # Normalize X
    x = (x - x_mean) / x_norm
    # add bias term
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    converted_x = np.matmul(x, M)

    converted_x = converted_x * y_norm + y_mean

    return converted_x

def test_fastl2lir_div(model_store, x, chunk_axis=1):
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

    x = (x - x_mean) / x_norm

    # Prediction
    y_pred_list = []
    for i, (Wf, bf) in enumerate(zip(W_files, b_files)):
        print('Chunk %d' % i)

        start_time = time()
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

        print('Elapsed time: %f s' % (time() - start_time))

    return np.concatenate(y_pred_list, axis=chunk_axis)


# Entry point ################################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()

'''Feature prediction with neural code conversion: prediction (test) script'''