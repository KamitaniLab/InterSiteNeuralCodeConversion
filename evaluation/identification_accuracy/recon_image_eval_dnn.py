from glob import glob
from itertools import product
import os
import itertools
import copy

import numpy as np
import pandas as pd
from PIL import Image
import torch

from bdpy.evals.metrics import pairwise_identification
from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.models import VGG19, AlexNet, layer_map

# Main #######################################################################

def recon_image_eval_dnn(
        recon_image_dir,
        true_image_dir,
        subjects=[], rois=[],
        recon_image_ext='tiff',
        true_image_ext='JPEG',
        recon_eval_encoder='AlexNet',
        device='cuda:0'
):
    '''Reconstruction evaluation with DNN features.'''

    # Display information
    print(f'Subjects: {subjects}')
    print(f'ROIs: {rois}\n')
    print(f'Reconstructed image dir: {recon_image_dir}')
    print(f'True images dir: {true_image_dir}\n')
    print(f'Evaluation encoder: {recon_eval_encoder}\n')

    # Loading data ###########################################################

    # Get recon image size
    reference_recon_image = glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*.' + recon_image_ext))[0]
    if not os.path.exists(reference_recon_image):
        raise RuntimeError("Not found:", reference_recon_image)

    img = Image.open(reference_recon_image)
    recon_image_size = img.size

    # True images
    true_image_files = sorted(glob(os.path.join(true_image_dir, '*.' + true_image_ext)))
    if len(true_image_files) == 0:
        raise RuntimeError("Not found true images:", os.path.join(true_image_dir, '*.' + true_image_ext))

    # Load true images
    true_images = [
        Image.open(f).convert("RGB").resize(recon_image_size, Image.LANCZOS)
        for f in true_image_files
    ]

    # Load DNN for metrics on DNN layer
    dnnh = DNNHandler(recon_eval_encoder, device=device)

    true_feat = dnnh.get_activation(true_images, flat=True)

    # Evaluating reconstruction performances ################################

    result_data = []

    for subject, roi in product(subjects, rois):
        print(f'DNN: {recon_eval_encoder} - Subject: {subject} - ROI: {roi}')

        recon_image_files = sorted(glob(os.path.join(
            recon_image_dir, subject, roi, '*.' + recon_image_ext
        )))
        recon_images = [
            Image.open(f).convert("RGB")  # No need to resize
            for f in recon_image_files
        ]

        recon_feat = dnnh.get_activation(recon_images, flat=True)

        for layer in dnnh.layers:
            ident_feat = pairwise_identification(recon_feat[layer], true_feat[layer])
            print(f"Layer: {layer}")
            print(f'Mean identification accuracy: {np.nanmean(ident_feat)}')

            result_data.append({
                'Layer': layer,
                'Identification accuracy': np.nanmean(ident_feat)
            })

    return result_data


# Class definitions ################################################################

class DNNHandler:
    """
    DNN quick handler (only forwarding)
    - AlexNet
    - VGG19
    """
    def __init__(self, encoder_name="AlexNet", device='cpu'):
        """Initialize the handler

        Parameters
        ----------
        encoder_name : str
            Specify the encoder name for the evaluation
            "AlexNet" or "VGG19".
        device : str
            Specify the machine environment.
            "cpu" or "cuda:0".
        """
        self.encoder_name = encoder_name
        self.device = device

        if encoder_name == "AlexNet":
            self.encoder = AlexNet()
            encoder_param_file = '/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt'
            self.encoder.to(device)
            self.encoder.load_state_dict(torch.load(encoder_param_file))
            self.encoder.eval()

            # AlexNet input image size (lab's specific value)
            self.image_size = [227, 227]
            # Mean of training images (ILSVRC2021_Training)
            self.mean_image = np.float32([104., 117., 123.])

            self.layer_mapping = layer_map("alexnet")

        elif encoder_name == "VGG19":
            self.encoder = VGG19()
            encoder_param_file = '/home/kiss/data/models_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'
            self.encoder.to(device)
            self.encoder.load_state_dict(torch.load(encoder_param_file))
            self.encoder.eval()

            # VGG19 input image size
            self.image_size = [224, 224]
            # Mean of training images (ILSVRC2021_Training)
            self.mean_image = np.float32([104., 117., 123.])

            self.layer_mapping = layer_map("vgg19")

        else:
            raise RuntimeError(f"This DNN is not implemented in dnn_evaluator: {self.encoder_name}")

        self.layers = list(self.layer_mapping.keys())
        self.feature_extractor = FeatureExtractor(self.encoder, self.layers, self.layer_mapping, device=self.device, detach=True)

    def get_activation(self, img_obj_list, flat=False):
        '''Obtain unit activation matrix

        Parameters
        ----------
        img_obj_list : list
            The list of PIL.Image object. (OR array objects)
        flat : bool (default: False)
            If True, the extracted feature is flatten in each image.

        Returns
        -------
        dictionary
            Dict object with the extracted features.
                {
                    "layer_name": ndarray <n samples x m units> or <n samples x m channels x h units x w units >
                }
        '''
        _img_obj_list = copy.deepcopy(img_obj_list)

        if not isinstance(_img_obj_list, list):
            _img_obj_list = [_img_obj_list]
        if isinstance(_img_obj_list[0], np.ndarray):
            _img_obj_list = [Image.fromarray(a_img) for a_img in _img_obj_list]

        activations = {layer: [] for layer in self.layers}
        for a_img in _img_obj_list:
            # Resize
            a_img = a_img.resize(self.image_size, Image.LANCZOS)
            x = np.asarray(a_img)

            # DNN specific preprocessing
            if self.encoder_name in ["AlexNet", "VGG19"]:
                # Swap dimensions and colour channels
                x = np.transpose(x, (2, 0, 1))[::-1]
                # Normalization (subtract the mean image)
                x = np.float32(x) - np.reshape(self.mean_image, (3, 1, 1))

            # Get activations
            features = self.feature_extractor.run(x)
            for layer in self.layers:
                activations[layer].append(features[layer])

        # Arrange
        for layer in self.layers:
            activations[layer] = np.vstack(activations[layer])
            if flat:
                activations[layer] = activations[layer].reshape(activations[layer].shape[0], -1)

        return activations


def run_evaluation(method, recon_base_dir, output_filename):
    true_image_dir = '../../data/test_image/source/'
    subjects_list = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05']
    subject = ['target']
    roi = ['VC']
    recon_eval_encoder = 'AlexNet'

    all_results = []

    for src, trg in itertools.permutations(subjects_list, 2):
        conversion = f'{src}_2_{trg}'
        recon_image_dir = os.path.join(recon_base_dir, conversion)

        result_data = recon_image_eval_dnn(
            recon_image_dir,
            true_image_dir,
            subjects=subject,
            rois=roi,
            recon_eval_encoder=recon_eval_encoder
        )

        for result in result_data:
            result.update({
                'Source': src,
                'Target': trg,
                'Method': method,
                'ROI': roi[0]
            })

        all_results.extend(result_data)

    data_df = pd.DataFrame(all_results)
    output_dir = './results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_df.to_csv(os.path.join(output_dir, output_filename), index=None)


# Entry point ################################################################

if __name__ == '__main__':
    run_evaluation(
        'content_loss',
        '../../reconstruction/content_loss/reconstruction',
        'dnn_identification_content_loss.csv'
    )

    run_evaluation(
        'brain_loss',
        '../../reconstruction/brain_loss/reconstruction',
        'dnn_identification_brain_loss.csv'
    )
