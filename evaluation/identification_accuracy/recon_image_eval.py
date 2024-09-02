from glob import glob
from itertools import product
import os, itertools

from bdpy.evals.metrics import pattern_correlation, pairwise_identification
import numpy as np
import pandas as pd
from PIL import Image

# Main #######################################################################

def recon_image_eval(
        recon_image_dir,
        true_image_dir,
        condition=[],
        output_file='./quality.pkl.gz',
        subjects=[], rois=[],
        recon_image_ext='tiff',
        true_image_ext='JPEG',
):

    # Display information
    print(f'Subjects: {subjects}')
    print(f'ROIs: {rois}\n')
    print(f'Reconstructed image dir: {recon_image_dir}')
    print(f'True images dir: {true_image_dir}\n')

    # Loading data ###########################################################

    # Get recon image size
    img = Image.open(glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*.' + recon_image_ext))[0])
    recon_image_size = img.size

    # True images
    true_image_files = sorted(glob(os.path.join(true_image_dir, '*.' + true_image_ext)))
    true_image_labels = [
        os.path.splitext(os.path.basename(a))[0]
        for a in true_image_files
    ]

    true_images = np.vstack([
        np.array(Image.open(f).resize(recon_image_size)).flatten()
        for f in true_image_files
    ])

    for subject, roi in product(subjects, rois):
        print(f'Subject: {subject} - ROI: {roi}')

        recon_image_files = sorted(glob(os.path.join(
            recon_image_dir, subject, roi, '*.' + recon_image_ext
        )))
        recon_image_labels = [
            os.path.splitext(os.path.basename(a))[0]
            for a in recon_image_files
        ]

        # Matching true and reconstructed images
        if len(recon_image_files) != len(true_image_files):
            raise RuntimeError(f'The number of true ({len(true_image_files)}) and reconstructed ({len(recon_image_files)}) images mismatch')

        for tf, rf in zip(true_image_labels, recon_image_labels):
            if tf not in rf:
                raise RuntimeError(f'Reconstructed image for {tf} not found')

        # Load reconstructed images
        recon_images = np.vstack([
            np.array(Image.open(f)).flatten()
            for f in recon_image_files
        ])

        # Calculate evaluation metrics
        r_pixelt = pattern_correlation(recon_images, true_images)
        ident = pairwise_identification(recon_images, true_images)

        print(f'Mean pixel correlation: {np.nanmean(r_pixelt)}')
        print(f'Mean identification accuracy: {np.nanmean(ident)}')

    return np.nanmean(r_pixelt), np.nanmean(ident)


def run_image_evaluation(method, recon_base_dir, output_filename):
    true_image_dir = '../../data/test_image/source/'
    subject = ['target']
    roi = ['VC']
    result_data = []
    subjects_list = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05']

    for src, trg in itertools.permutations(subjects_list, 2):
        conversion = f'{src}_2_{trg}'
        recon_image_dir = os.path.join(recon_base_dir, conversion)

        r_pixel, ident = recon_image_eval(
            recon_image_dir,
            true_image_dir,
            subjects=subject,
            rois=roi
        )
        result_data.append({
            'Source': src,
            'Target': trg,
            'Identification accuracy': ident,
            'Method': method,
            'ROI': roi[0]
        })

    data_df = pd.DataFrame(result_data)
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

