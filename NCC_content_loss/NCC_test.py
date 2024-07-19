#!/usr/bin/python3

import argparse
import os
from torch.autograd import Variable
import torch
from time import time
from models import Converter
import numpy as np
from bdpy.dataform import Features, load_array, save_array
from bdpy.util import makedir_ifnot, get_refdata
from utils import fastl2lir_parameter,test_fastl2lir_revise
import itertools,bdpy
from utils import PathBuilder

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=50, help='size of the batches')
parser.add_argument('--number_size', type=int, default=6000, help='size of the samples')
parser.add_argument('--size', type=int, default=64, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

print(opt)

if torch.cuda.is_available() :
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

brain_dir = '/home/nu/hbwang/data/fmri_shared/datasets/Deeprecon/fmriprep'
subjects_list = {'sub01': 'sub-01_NaturalImageTest.h5',
                 'sub02': 'sub-02_NaturalImageTest.h5',
                 # 'sub03': 'sub-03_NaturalImageTest.h5',
                 # 'sub04': 'sub-04_NaturalImageTest.h5',
                 # 'sub05': 'sub-05_NaturalImageTest.h5'
                 }

data_brain = {subject: bdpy.BData(os.path.join(brain_dir, dat_file))
              for subject, dat_file in subjects_list.items()}

vgg_dir = '/home/kiss/data/contents_shared/ImageNetTest/derivatives/features'
vgg_network = 'caffe/VGG_ILSVRC_19_layers'

src_decoder_dir = '../srcdecoder_dir'
trg_decoder_dir = '../trgdecoder_dir'

label_name = 'image_index'
rois_list = {
    'VC': 'ROI_VC =1',
    # 'V1': 'ROI_V1 = 1',
    # 'V2': 'ROI_V2 = 1',
    # 'V3': 'ROI_V3 = 1',
    # 'V4': 'ROI_hV4 = 1',
    # 'HVC': 'ROI_HVC = 1'
}

for src,trg in itertools.permutations(subjects_list.keys(),2):
    for roi in rois_list:
        subject_src = src
        subject_trg = trg

        conversion = src + '_2_' + trg
        print('Source: %s' %  subject_src)
        print('Target: %s' %  subject_trg)
        print('Source: %s' %  conversion)
        print('ROI: %s' % roi)

        x = data_brain[subject_src].select(rois_list[roi])
        x_labels = data_brain[subject_src].select('image_index') # src_brain data
        input_nc = x.shape[1]

        path_src = PathBuilder(src_decoder_dir, vgg_network, subject_src, roi)
        path_trg = PathBuilder(trg_decoder_dir, vgg_network, subject_trg, roi)

        x_mean_src, x_norm_src, _, _, _, _ = fastl2lir_parameter(path_src.build_model_path('fc6'), chunk_axis=1)
        x_mean_trg,  x_norm_trg, _, _, _, _ = fastl2lir_parameter(path_trg.build_model_path('fc6'), chunk_axis=1)

        print(x_mean_trg.shape)
        output_nc = x_mean_trg.shape[1]

        # Loaded the trained model
        netG_A2B = Converter(input_nc, output_nc)

        # if opt.cuda:
        netG_A2B.cuda()

        # Load state dicts
        converter_dir = os.path.join('output', conversion)
        device = torch.device('cuda:0')
        netG_A2B.load_state_dict(torch.load(os.path.join(converter_dir,roi,'model.pth'),map_location=device))

        # Set model's test mode
        netG_A2B.eval()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor
        ones = Variable(Tensor(opt.batchSize, 1).fill_(1.0), requires_grad=False)
        # if opt.cuda else torch.Tensor

        x_test_labels_unique = np.unique(x_labels)
        x_averaged = np.vstack([np.mean(x[(np.array(x_labels) == lb).flatten(), :], axis=0) for lb in x_test_labels_unique])
        x_item = (x_averaged - x_mean_src)/x_norm_src
        real_A = Variable(Tensor(x_item),requires_grad=False)

        # Converte brain activity
        fake_B = netG_A2B(real_A)

        pred_B = fake_B.detach().cpu().numpy()
        y_pred = x_norm_trg * pred_B + x_mean_trg

        # Image features
        features_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                         'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                         'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                         'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                         'fc6', 'fc7', 'fc8'][::-1]

        # feature decoding
        for vgg_feat in features_list:

            start_time = time()
            pred_dnn = test_fastl2lir_revise(path_trg.build_model_path(vgg_feat), path_src.build_model_path(vgg_feat), y_pred)
            print('Total elapsed time (prediction): %f' % (time() - start_time))
            print('VGG_feature: %s' % vgg_feat)

            results_dir_root = './result_vgg'
            results_dir_prediction = os.path.join(results_dir_root, conversion, vgg_network, vgg_feat, "target", roi)

            makedir_ifnot(results_dir_prediction)
            for i, label in enumerate(x_test_labels_unique):
                # Predicted features
                feature = np.array([pred_dnn[i,]])  # To make feat shape 1 x M x N x ...

                # Save file name
                save_file = os.path.join(results_dir_prediction, '%s.mat' % label)

                # Save
                save_array(save_file, feature, 'feat', dtype=np.float32, sparse=False)

            print('Saved %s' % results_dir_prediction)












