from itertools import product
import os
import itertools

from bdpy.dataform import Features, DecodedFeatures
from bdpy.evals.metrics import profile_correlation, pattern_correlation
import hdf5storage
import numpy as np
import pandas as pd

# Main #######################################################################

def featdec_eval(
        decoded_feature_dir,
        true_feature_dir,
        output_file='./accuracy.pkl.gz',
        subjects=None,
        subject_list=None,
        rois=None,
        method=None,
        network=None,
        features=None,
        feature_index_file=None,
        feature_decoder_dir=None,
        single_trial=False
):
    '''Evaluation of feature decoding.

    Input:

    - decoded_feature_dir
    - true_feature_dir

    Output:

    - output_file

    Parameters:

    TBA
    '''

    # Display information
    print(f'Subjects: {subjects}')
    print(f'ROIs: {rois}\n')
    print(f'Decoded features: {decoded_feature_dir}\n')
    print(f'True features (Test): {true_feature_dir}\n')
    print(f'Layers: {features}\n')
    if feature_index_file is not None:
        print(f'Feature index: {feature_index_file}\n')

    # Loading data ###########################################################

    # True features
    features_test = Features(true_feature_dir, feature_index=feature_index_file) if feature_index_file else Features(true_feature_dir)

    # Evaluating decoding performances #######################################
    perf_df = []

    for src, trg in itertools.permutations(subject_list, 2):

        # Decoded features
        conversion = f'{src}_2_{trg}'
        decoded_features = DecodedFeatures(os.path.join(decoded_feature_dir, conversion, network))

        print(os.path.join(decoded_feature_dir, conversion, network))
        for layer in features:
            print(f'Layer: {layer}')

            for subject, roi in product(subjects, rois):
                print(f'Subject: {subject} - ROI: {roi}')

                pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi)

                pred_labels = [int(float(i)) for i in decoded_features.selected_label]

                true_y = features_test.get_features(layer)
                true_labels = features_test.index

                true_y_sorted = true_y[np.array([np.where(np.array(true_labels) == x)[0][0] for x in pred_labels])] if not np.array_equal(pred_labels, true_labels) else true_y

                # Load Y mean and SD
                norm_param_dir = os.path.join(feature_decoder_dir, layer, trg, roi, 'model')
                train_y_mean = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
                train_y_std = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

                pred_y = pred_y.reshape([pred_y.shape[0], -1])
                true_y_sorted = true_y_sorted.reshape([true_y_sorted.shape[0], -1])

                r_prof = profile_correlation(pred_y, true_y_sorted)
                r_patt = pattern_correlation(pred_y, true_y_sorted, mean=train_y_mean, std=train_y_std)

                print(f'Mean profile correlation: {np.nanmean(r_prof)}')
                print(f'Mean pattern correlation: {np.nanmean(r_patt)}')

                perf_df.append({
                    'Source': src, 'Target': trg,
                    'Profile correlation': np.nanmean(r_prof.flatten()),
                    'Pattern correlation': np.nanmean(r_patt.flatten()),
                    'Method': method, 'ROI': roi, 'Layer': layer
                })

    print(perf_df)

    data_df = pd.DataFrame(perf_df)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_df.to_csv(output_file, index=None)
    return output_file


def run_evaluation(method, decoded_feature_dir, output_filename):
    network = 'caffe/VGG_ILSVRC_19_layers'
    test_feature_dir = '../../data/test_image/true_features/contents_shared/ImageNetTest/derivatives/features'
    output_dir = './results'
    features_list = [
        'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
        'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
        'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
        'fc6', 'fc7', 'fc8'][::-1]

    subject = ['target']
    roi = ['VC']
    conversion_list = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05']

    feature_decoder_dir = os.path.join(
        '../../data/pre-trained/decoders/ImageNetTraining/deeprecon_pyfastl2lir_alpha100_vgg19_allunits', network)

    featdec_eval(
        decoded_feature_dir,
        os.path.join(test_feature_dir, network),
        output_file=os.path.join(output_dir, output_filename),
        subjects=subject,
        subject_list=conversion_list,
        rois=roi,
        method=method,
        network=network,
        features=features_list,
        feature_decoder_dir=feature_decoder_dir,
    )

if __name__ == '__main__':
    run_evaluation('content_loss', '../../NCC_content_loss/result_vgg', 'decoding_accuracy_content_loss.csv')
    run_evaluation('brain_loss', '../../NCC_brain_loss/result_vgg', 'decoding_accuracy_brain_loss.csv')