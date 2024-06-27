import bdpy
import numpy as np
import pandas as pd
import hdf5storage 
import matplotlib.pyplot as plt
from collections import OrderedDict 
import os

converter_param = './params/converter_params_6000_VC.csv'
df_param = pd.read_csv(converter_param)
brain_dir = '../fmridata'
subjects_list = {'sub01': 'sub-01_NaturalImageTest.h5',
                 'sub02': 'sub-02_NaturalImageTest.h5',
                 'sub03': 'sub-03_NaturalImageTest.h5',
                 'sub04': 'sub-04_NaturalImageTest.h5',
                 'sub05': 'sub-05_NaturalImageTest.h5'}

data_brain = {subject: bdpy.BData(os.path.join(brain_dir, dat_file))
            for subject, dat_file in subjects_list.items()}

label_name = 'image_index'

nc_models_dir_root = os.path.join('./NCconverter_results','ncc_training')

output_filename = 'conversion_accuracy_profile'

base_ROI = 'ROI_VC'
rep = 24


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
    for roi,roi_str in roi_mapping.items():
        print(roi)
        
        _, idx = data.select(roi_str, return_index=True)
        
        loc = np.where(idx)[0]
        idxs[roi] = [idx_mapper[l] for l in loc]

    return idxs


roi_dict = {
                 'VC': 'ROI_VC',
                  'V1': 'ROI_V1',
                  'V2': 'ROI_V2',
                  'V3': 'ROI_V3',
                  'V4': 'ROI_hV4',
                  'HVC': 'ROI_HVC'

    }


result_data = []

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

    dat1 = data_brain[src]
    dat2 = data_brain[trg]
    
    x_test, base_idx_src = dat1.select(base_ROI, return_index=True)
    y_test, base_idx_trg = dat2.select(base_ROI, return_index=True)

    x_test_labels = dat1.select('image_index')
    y_test_labels = dat2.select('image_index')

    x_test_labels_unique = np.unique(x_test_labels)
    y_test_labels_unique = np.unique(y_test_labels)

    x = np.vstack([x_test[(x_test_labels == lb).flatten(), :] for lb in x_test_labels_unique])
    y = np.vstack([y_test[(y_test_labels == lb).flatten(), :] for lb in y_test_labels_unique])


    conversion = src+'_2_'+trg
    nc_model_dir = os.path.join(nc_models_dir_root, conversion, roi, method, 
                       str(num_samples), 'model')

    converted_x = test_ncconverter(nc_model_dir, x)


    y_roi_idxs = compute_roi_index(dat2, base_ROI, roi_dict)
    for trg_roi, roi_str in roi_dict.items():
        y_sub = y[:, y_roi_idxs[trg_roi]]
        converted_x_sub = converted_x[:,y_roi_idxs[trg_roi]]

        for i in range(y_sub.shape[1]):
            y_sub_vox = y_sub[:,i].reshape(rep,-1,order='F')
            converted_x_sub_vox = converted_x_sub[:,i].reshape(rep,-1,order='F')
            corr = np.mean(np.corrcoef(y_sub_vox,converted_x_sub_vox)[rep:,:rep])

            result_data.append({'Source': src, 'Target':trg, 'Number of samples': num_samples,
                                'Correlation': corr, 'Method': 'NCC', 'ROI':trg_roi, 'Vox_idx': y_roi_idxs[trg_roi][i], 'Target ROI': trg_roi})

output_dir = './results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
df = pd.DataFrame(result_data)
df.to_csv(os.path.join(output_dir, output_filename), index=None)


