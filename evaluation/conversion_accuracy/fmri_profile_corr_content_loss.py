import os
import sys
import itertools
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import bdpy

# Adding the parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from NCC_content_loss.models import Converter
from NCC_content_loss.utils import PathBuilder, fastl2lir_parameter

def load_brain_data(brain_dir, subjects_list):
    """
    Load brain data for each subject from the specified directory.
    """
    return {subject: bdpy.BData(os.path.join(brain_dir, dat_file)) for subject, dat_file in subjects_list.items()}

def compute_roi_index(data, embeded_roi, roi_mapping):
    """
    Compute the ROI indices to aggregate conversion matrices for each brain area.
    """
    _, base_idx = data.select(embeded_roi, return_index=True)
    idx_loc = np.where(base_idx)[0]
    idx_mapper = dict(zip(idx_loc, range(len(idx_loc))))

    idxs = {}
    for roi, roi_str in roi_mapping.items():
        _, idx = data.select(roi_str, return_index=True)
        loc = np.where(idx)[0]
        idxs[roi] = [idx_mapper[l] for l in loc]

    return idxs

def initialize_model(input_nc, output_nc, model_path):
    """
    Initialize and load the conversion model.
    """
    netG_A2B = Converter(input_nc, output_nc)
    netG_A2B.cuda()
    netG_A2B.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
    return netG_A2B

def process_subject_pair(src, trg, data_brain, roi_dict, result_data, base_ROI, pre_vgg_models_dir_root, vgg_network, rep, num_samples=6000):
    """
    Process a pair of subjects, perform conversion, and calculate correlation.
    """
    dat1 = data_brain[src]
    dat2 = data_brain[trg]

    x_test, _ = dat1.select(base_ROI, return_index=True)
    y_test, _ = dat2.select(base_ROI, return_index=True)
    input_nc, output_nc = x_test.shape[1], y_test.shape[1]

    conversion = f'{src}_2_{trg}'
    print(f'Source: {src}, Target: {trg}')

    # Initialize and load the conversion model
    model_path = os.path.join(
        '../../NCC_content_loss/converters', conversion, 'VC',
        'model.pth')
    netG_A2B = initialize_model(input_nc, output_nc, model_path)

    # Extract test labels and unique labels
    x_test_labels = dat1.select('image_index')
    y_test_labels = dat2.select('image_index')
    x_test_labels_unique = np.unique(x_test_labels)
    y_test_labels_unique = np.unique(y_test_labels)

    # Combine data by unique labels
    x = np.vstack([x_test[(x_test_labels == lb).flatten(), :] for lb in x_test_labels_unique])
    y = np.vstack([y_test[(y_test_labels == lb).flatten(), :] for lb in y_test_labels_unique])

    # Load and normalize feature decoder parameters
    path_src = PathBuilder(pre_vgg_models_dir_root, vgg_network, src, "VC")
    path_trg = PathBuilder(pre_vgg_models_dir_root, vgg_network, trg, "VC")
    x_mean_src, x_norm_src = fastl2lir_parameter(path_src.build_model_path('fc6'), chunk_axis=1)[:2]
    x_mean_trg, x_norm_trg = fastl2lir_parameter(path_trg.build_model_path('fc6'), chunk_axis=1)[:2]

    x_item = (x - x_mean_src) / x_norm_src
    real_A = Variable(torch.cuda.FloatTensor(x_item), requires_grad=False)

    # Generate output
    fake_B = netG_A2B(real_A).detach().cpu().numpy()
    converted_x = x_norm_trg * fake_B + x_mean_trg

    # Compute ROI indices and calculate correlations
    y_roi_idxs = compute_roi_index(dat2, base_ROI, roi_dict)
    calculate_correlations(y, converted_x, y_roi_idxs, src, trg, result_data, rep, num_samples)

def calculate_correlations(y, converted_x, y_roi_idxs, src, trg, result_data, rep, num_samples):
    """
    Calculate correlations for each voxel and store the results.
    """
    for trg_roi, roi_idxs in y_roi_idxs.items():
        y_sub = y[:, roi_idxs]
        converted_x_sub = converted_x[:, roi_idxs]

        for i in range(y_sub.shape[1]):
            y_sub_vox = y_sub[:, i].reshape(rep, -1, order='F')
            converted_x_sub_vox = converted_x_sub[:, i].reshape(rep, -1, order='F')
            corr = np.mean(np.corrcoef(y_sub_vox, converted_x_sub_vox)[rep:, :rep])

            result_data.append({
                'Source': src,
                'Target': trg,
                'Number of samples': num_samples,
                'Correlation': corr,
                'Method': 'content_loss',
                'ROI': trg_roi,
                'Vox_idx': i,
                'Target ROI': trg_roi
            })

def save_results(result_data, output_dir, output_filename):
    """
    Save the results to a CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(result_data)
    df.to_csv(os.path.join(output_dir, output_filename), index=None)
    print(f'Results saved to {os.path.join(output_dir, output_filename)}')

def main():
    """
    Main function to execute the process for all subject pairs and save the results.
    """
    # Constants and paths
    brain_dir = '../../data/fmri'
    vgg_network = 'caffe/VGG_ILSVRC_19_layers'
    pre_vgg_models_dir_root = '../../data/feature_decoders/ImageNetTraining/deeprecon_pyfastl2lir_alpha100_vgg19_allunits'
    output_dir = './results'
    output_filename = 'conversion_accuracy_profile_content_loss.csv'
    base_ROI = 'ROI_VC'
    rep = 24

    # Subjects list and ROI dictionary
    subjects_list = {
        'sub01': 'sub-01_NaturalImageTest.h5',
        'sub02': 'sub-02_NaturalImageTest.h5',
        'sub03': 'sub-03_NaturalImageTest.h5',
        'sub04': 'sub-04_NaturalImageTest.h5',
        'sub05': 'sub-05_NaturalImageTest.h5'
    }
    roi_dict = {
        'VC': 'ROI_VC',
        'V1': 'ROI_V1',
        'V2': 'ROI_V2',
        'V3': 'ROI_V3',
        'V4': 'ROI_hV4',
        'HVC': 'ROI_HVC'
    }

    data_brain = load_brain_data(brain_dir, subjects_list)
    result_data = []

    for src, trg in itertools.permutations(subjects_list.keys(), 2):
        process_subject_pair(src, trg, data_brain, roi_dict, result_data, base_ROI, pre_vgg_models_dir_root, vgg_network, rep)

    save_results(result_data, output_dir, output_filename)

if __name__ == "__main__":
    main()
