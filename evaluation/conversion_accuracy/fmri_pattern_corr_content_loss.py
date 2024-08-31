import os
import sys
import itertools
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import bdpy
from itertools import product

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
    Compute ROI indices for each brain area to aggregate conversion matrices.
    """
    _, base_idx = data.select(embeded_roi, return_index=True)
    idx_loc = np.where(base_idx)[0]
    idx_mapper = {loc: i for i, loc in enumerate(idx_loc)}

    return {
        roi: [idx_mapper[l] for l in np.where(data.select(roi_str, return_index=True)[1])[0]]
        for roi, roi_str in roi_mapping.items()
    }


def initialize_model(input_nc, output_nc, model_path):
    """
    Initialize and load the conversion model.
    """
    model = Converter(input_nc, output_nc)
    model.cuda()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
    return model


def normalize_data(x, mean, norm):
    """
    Normalize the data using the provided mean and norm.
    """
    return (x - mean) / norm


def process_subject_pair(src, trg, data_brain, roi_dict, result_data, base_ROI, pre_vgg_models_dir_root, vgg_network, trials_per_img, num_samples=6000):
    """
    Process a pair of subjects, perform conversion, and calculate correlation.
    """
    dat1 = data_brain[src]
    dat2 = data_brain[trg]

    x_test, _ = dat1.select(base_ROI, return_index=True)
    y_test, _ = dat2.select(base_ROI, return_index=True)
    input_nc, output_nc = x_test.shape[1], y_test.shape[1]

    conversion = f'{src}_2_{trg}'
    print(f'Source: {src}')
    print(f'Target: {trg}')

    # Initialize and load the conversion model
    model_path = os.path.join(
        '../../NCC_content_loss/converters', conversion, 'VC',
        'model.pth')
    netG_A2B = initialize_model(input_nc, output_nc, model_path)

    # Extract test labels
    x_test_labels = dat1.select('image_index')
    y_test_labels = dat2.select('image_index')

    # Load and normalize the feature decoder parameters
    path_src = PathBuilder(pre_vgg_models_dir_root, vgg_network, src, "VC")
    path_trg = PathBuilder(pre_vgg_models_dir_root, vgg_network, trg, "VC")
    x_mean_src, x_norm_src = fastl2lir_parameter(path_src.build_model_path('fc6'), chunk_axis=1)[:2]
    x_mean_trg, x_norm_trg = fastl2lir_parameter(path_trg.build_model_path('fc6'), chunk_axis=1)[:2]

    x_item = normalize_data(x_test, x_mean_src, x_norm_src)
    real_A = Variable(torch.cuda.FloatTensor(x_item), requires_grad=False)

    # Generate output
    fake_B = netG_A2B(real_A)
    converted_x = fake_B.detach().cpu().numpy()
    y = normalize_data(y_test, x_mean_trg, x_norm_trg)

    y_roi_idxs = compute_roi_index(dat2, base_ROI, roi_dict)
    calculate_correlations(x_test_labels, y_test_labels, y_roi_idxs, y, converted_x, result_data, src, trg, trials_per_img, num_samples)


def calculate_correlations(x_test_labels, y_test_labels, y_roi_idxs, y, converted_x, result_data, src, trg, trials_per_img, num_samples):
    """
    Calculate the correlations between the converted and original data.
    """
    for trg_roi, roi_idxs in y_roi_idxs.items():
        y_sub = y[:, roi_idxs]
        converted_x_sub = converted_x[:, roi_idxs]

        for image_index in np.unique(x_test_labels):
            converted_x_block = converted_x_sub[(x_test_labels == image_index).flatten(), :]
            y_block = y_sub[(y_test_labels == image_index).flatten(), :]

            corr_block = [
                np.corrcoef(y_block[m, :], converted_x_block[n, :])[0, 1]
                for m, n in product(range(trials_per_img), repeat=2)
            ]

            corr_mean = np.mean(corr_block)
            print(f'{src} -> {trg}, ROI: {trg_roi}, Image Index: {image_index}, Correlation: {corr_mean}')

            result_data.append({
                'Source': src,
                'Target': trg,
                'Number of samples': num_samples,
                'Correlation': corr_mean,
                'Method': 'Ours',
                'ROI': trg_roi,
                'Image index': image_index
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
    # Directory paths and settings
    brain_dir = '../../data/fmri'
    vgg_network = 'caffe/VGG_ILSVRC_19_layers'
    pre_vgg_models_dir_root = '../../data/feature_decoders/ImageNetTraining/deeprecon_pyfastl2lir_alpha100_vgg19_allunits'
    output_dir = './results'
    output_filename = 'conversion_accuracy_pattern_content_loss.csv'

    # Constants
    base_ROI = 'ROI_VC'
    trials_per_img = 24

    # Subjects list
    subjects_list = {
        'sub01': 'sub-01_NaturalImageTest.h5',
        'sub02': 'sub-02_NaturalImageTest.h5',
        'sub03': 'sub-03_NaturalImageTest.h5',
        'sub04': 'sub-04_NaturalImageTest.h5',
        'sub05': 'sub-05_NaturalImageTest.h5'
    }

    # ROI dictionary
    roi_dict = {
        'VC': 'ROI_VC',
        'V1': 'ROI_V1',
        'V2': 'ROI_V2',
        'V3': 'ROI_V3',
        'V4': 'ROI_hV4',
        'HVC': 'ROI_HVC'
    }

    # Load brain data
    data_brain = load_brain_data(brain_dir, subjects_list)
    result_data = []

    # Process each pair of subjects
    for src, trg in itertools.permutations(subjects_list.keys(), 2):
        process_subject_pair(src, trg, data_brain, roi_dict, result_data, base_ROI, pre_vgg_models_dir_root, vgg_network, trials_per_img)

    # Save the results
    save_results(result_data, output_dir, output_filename)

if __name__ == "__main__":
    main()
