#!/usr/bin/python3
#this one
import argparse
import itertools
from torch.autograd import Variable
import torch
import random
import concurrent
from concurrent.futures import ProcessPoolExecutor
from models import Generator
from utils_new import fastl2lir_parameter, dnn_chunk_get
from utils_new import LambdaLR,PathBuilder
from utils_new import Logger
from utils_new import test_fastl2lir_div,compute_layer_weights
import bdpy
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, default=0, help='starting iteration')
parser.add_argument('--n_iterations', type=int, default=1024, help='number of iterations of training')
parser.add_argument('--number_size', type=int, default=6000, help='size of the samples')
parser.add_argument('--batchSize', type=int, default=6000, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument('--decay_iteration', type=int, default=512, help='iteration to start linearly decaying the learning rate to 0')
parser.add_argument('--cuda', action='store_true', help='enable CUDA')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')

# Set the global Tensor type
Tensor = torch.FloatTensor

def select_samples(x, x_labels,path_src, num_sample):
    # First, sort the data
    x_index = np.argsort(x_labels.flatten())
    x_labels = x_labels[x_index]
    x = x[x_index, :]

    # If only a sample size of less than 1200 is required, we choose the first replicate
    if num_sample < 1200:
        rep = 1
    else:
        rep = int(num_sample / 1200)

    # 选择需要的重复
    tmp = np.zeros(5, dtype=bool)
    tmp[:rep] = True
    sel = np.tile(tmp, 1200)
    x_labels = x_labels[sel]
    x = x[sel]

    # If you only need a sample size of less than 1200, choose different categories of samples to avoid bias
    # Here we have 150 image categories with 8 images per category

    if num_sample == 300:
        # 2 images per category
        x = x[0::4]
        x_labels = x_labels[0::4]
    elif num_sample == 600:
        # 4 images per class
        x = np.vstack((x[0::4], x[1::4]))
        x_labels = np.vstack((x_labels[0::4], x_labels[1::4]))

    elif num_sample == 900:
        # 6 images per category
        x = np.vstack((x[0::4], x[1::4], x[2::4]))
        x_labels = np.vstack((x_labels[0::4], x_labels[1::4], x_labels[2::4]))

    # Normalize
    src_mean_std_dir = path_src.build_model_path('fc6')
    x_mean_src, x_norm_src, _, _, _, _ = fastl2lir_parameter(src_mean_std_dir, chunk_axis=1)
    x = (x - x_mean_src) / x_norm_src

    return x, x_labels


def process_layer(layer, path_trg, dnn_index, chunk_dir, current_indices):
    dnn_chunk_path = os.path.join(chunk_dir, layer)
    dnn = dnn_chunk_get(dnn_chunk_path, current_indices[layer])

    dnn_mean_std_dir = path_trg.build_model_path(layer)
    _, _, y_mean_trg, y_norm_trg, _, _ = fastl2lir_parameter(dnn_mean_std_dir, chunk_axis=1)

    if "conv" in layer:
        layer_features = (dnn[dnn_index] - y_mean_trg[:, current_indices[layer], :, :]) / y_norm_trg[:, current_indices[layer], :, :]
    elif "fc" in layer:
        layer_features = (dnn[dnn_index] - y_mean_trg) / y_norm_trg

    return layer, layer_features

def prepare_dnn_features(features_list, path_trg, dnn_index, chunk_dir, current_indices):

    all_dnn_features = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_layer, layer, path_trg, dnn_index, chunk_dir, current_indices) for layer in features_list]
        for future in concurrent.futures.as_completed(futures):
            layer, layer_features = future.result()
            all_dnn_features[layer] = layer_features
            # print(layer, layer_features.shape)
    return all_dnn_features

# initialize_network_and_optimizer
def initialize_network_and_optimizer(input_nc, output_nc, opt):
    model = Generator(input_nc, output_nc)
    if opt.cuda and torch.cuda.is_available():
        model.cuda()
    print(model)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_iterations, opt.iteration, opt.decay_iteration).step)
    return model, optimizer, lr_scheduler

# save the model
def save_model(model, save_path):
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(model.state_dict(), save_path)

def prepare_indices(num_iterations, features_list):
    chunks_number = {
        'conv1_1': 64, 'conv1_2': 64,
        'conv2_1': 128, 'conv2_2': 128,
        'conv3_1': 256, 'conv3_2': 256, 'conv3_3': 256, 'conv3_4': 256,
        'conv4_1': 512, 'conv4_2': 512, 'conv4_3': 512, 'conv4_4': 512,
        'conv5_1': 512, 'conv5_2': 512, 'conv5_3': 512, 'conv5_4': 512,
        'fc1': 4096, 'fc2': 4096, 'fc3': 1000
    }
    prepared_indices = {}
    for layer in features_list:
        if "fc" in layer:
            # For the fully connected layer, the index is all zeros
            prepared_indices[layer] = [0] * num_iterations
        elif layer in chunks_number:
            # For convolutional layers, a random index is generated based on the number of channels in the layer
            current_indices = random.sample(range(chunks_number[layer]), chunks_number[layer])
            indices = []
            for i in range(num_iterations):
                if i % len(current_indices) == 0 and i > 0:
                    current_indices = random.sample(current_indices, len(current_indices))
                indices.append(current_indices[i % len(current_indices)])
            prepared_indices[layer] = indices
    return prepared_indices

def allocate_memory_for_layers(features_list, path_trg, dnn_index, chunk_dir, current_indices, opt):
    # Use prepare_dnn_features to get the shape of each layer
    initial_chunk_dnn_features = prepare_dnn_features(features_list, path_trg, dnn_index, chunk_dir, current_indices)

    # Pre-allocate memory
    input_dnn_layers = {}
    for layer in features_list:
        if "conv" in layer:
            shape = initial_chunk_dnn_features[layer].shape
            input_dnn_layers[layer] = Tensor(opt.batchSize, *shape[1:])
        elif "fc" in layer:
            shape = initial_chunk_dnn_features[layer].shape
            input_dnn_layers[layer] = Tensor(opt.batchSize, shape[1])

    return input_dnn_layers

def chunk_process_batch(model, real_A, chunk_dnn_features, input_dnn_layers, features_list, criterion_mse, path_trg, current_indices):
    # generate loss
    fake_B = model(real_A)

    # Initialize total loss for this batch
    total_loss = 0

    # calculate weight for different layers
    layer_weights = compute_layer_weights(chunk_dnn_features, features_list)

    for layer in features_list:
        # Get the i-th batch for the corresponding layer (B)
        dnn_feature_batch = chunk_dnn_features[layer]
        dnn_true = Variable(input_dnn_layers[layer].copy_(torch.from_numpy(dnn_feature_batch)))

        chunk_i = current_indices[layer]
        decoding_path = path_trg.build_model_path(layer)
        y_B = test_fastl2lir_div(decoding_path, fake_B, chunk_i)

        loss = criterion_mse(dnn_true, y_B.squeeze(1))

        # Accumulate loss for all layers
        total_loss += layer_weights[layer] * loss

    return total_loss

def converter_training(subject_src, subject_trg, data_brain, rois_list, roi, vgg_dir, features_list, src_network, trg_network, src_dir, trg_dir, chunks_index_dict, opt):

    conversion = f"{subject_src}_2_{subject_trg}"
    print(f"Conversion: {conversion}")

    x = data_brain[subject_src].select(rois_list[roi])
    x_labels = data_brain[subject_src].select('image_index')  # src_brain data

    ### The directory of mean and std
    path_src = PathBuilder(src_dir, src_network, subject_src, roi)
    path_trg = PathBuilder(trg_dir, trg_network, subject_trg, roi)

    ### load the dimension of target brain activities instead of target brain data
    x_mean_trg, _, _, _, _, _ = fastl2lir_parameter(path_trg.build_model_path('fc6'), chunk_axis=1)
    print(x_mean_trg.shape)
    input_nc = x.shape[1]
    output_nc = x_mean_trg.shape[1]

    # brain activity processing: select_samples and normarlized it
    x, brain_labels = select_samples(x, x_labels, path_src, opt.number_size)
    print(brain_labels.shape)

    # align labels of brain activity and dnn
    dnn_labels = np.unique(brain_labels)
    dnn_index = np.array([np.where(np.array(dnn_labels) == xl) for xl in brain_labels]).flatten()

    ###### Initialize thte neural network ######
    model, optimizer, lr_scheduler = initialize_network_and_optimizer(input_nc, output_nc, opt)
    criterion_mse = torch.nn.MSELoss(reduction='sum')

    # Inputs & targets memory allocation
    # Get the initial settings for the current index
    initial_indices = {layer: chunks_index_dict[layer][0] for layer in features_list}
    # Get the shape of each DNN layers
    input_dnn_layers = allocate_memory_for_layers(features_list, path_trg, dnn_index, vgg_dir, initial_indices, opt)

    real_A = Variable(Tensor(x), requires_grad=False)
    # # Loss plot
    logger = Logger(opt.n_iterations, 1)

    for iteration in range(opt.iteration, opt.n_iterations):

        current_indices = {layer: chunks_index_dict[layer][iteration] for layer in features_list}

        # DNN feature processing
        chunk_dnn_features = prepare_dnn_features(features_list, path_trg, dnn_index, vgg_dir, current_indices)

        optimizer.zero_grad()
        total_loss = chunk_process_batch(model, real_A, chunk_dnn_features, input_dnn_layers, features_list, criterion_mse, path_trg, current_indices)
        total_loss.backward() #Backward and optimize after accumulating losses from all layers
        optimizer.step()

        # Progress report for the batch
        logger.log({f'Deeprecon_{conversion}_': total_loss})
        # Update learning rates
        lr_scheduler.step()
        # Save models checkpoints
        if iteration == opt.n_iterations - 1:
            save_model_path = os.path.join('output', conversion, roi, 'model.pth')
            save_model(model, save_model_path)

    return model
##############################################################################################################
def main():
    global Tensor
    opt = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if opt.cuda and torch.cuda.is_available():
        torch.cuda.set_device(int(opt.gpu_id))
        torch.cuda.manual_seed_all(seed)
        Tensor = torch.cuda.FloatTensor

    else:
        Tensor = torch.FloatTensor

    # brain data
    brain_dir = '../data/fmri'
    subjects_list = {
        'sub01': 'sub-01_NaturalImageTraining.h5',
        'sub02': 'sub-02_NaturalImageTraining.h5',
        # 'sub03': 'sub-03_NaturalImageTraining.h5',
        # 'sub04': 'sub-04_NaturalImageTraining.h5',
        # 'sub05': 'sub-05_NaturalImageTraining.h5'
    }

    data_brain = {subject: bdpy.BData(os.path.join(brain_dir, dat_file))
                  for subject, dat_file in subjects_list.items()}
    rois_list = {
        'VC': 'ROI_VC =1',
        # 'V1': 'ROI_V1 = 1',
        # 'V2': 'ROI_V2 = 1',
        # 'V3': 'ROI_V3 = 1',
        # 'V4': 'ROI_hV4 = 1',
        # 'HVC': 'ROI_HVC = 1'
    }

    # The decoder of the directory
    src_decoder_dir = '../srcdecoder_dir'
    trg_decoder_dir = '../trgdecoder_dir'

    # dnn feature
    vgg_dir = ''

    src_network = 'caffe/VGG_ILSVRC_19_layers'
    trg_network = 'caffe/VGG_ILSVRC_19_layers'

    features_list = [   'conv1_1', 'conv1_2',
                        'conv2_1', 'conv2_2',
                        'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                        'conv5_1', 'conv5_2', 'conv5_3',
                        'conv5_4',
                        'fc6', 'fc7','fc8'][::-1]

    chunks_index_dict= prepare_indices(opt.n_iterations,features_list)

    for src, trg in itertools.permutations(subjects_list.keys(), 2):
        conversion = src + '_2_' + trg
        print('Source: %s' % src)
        print('Target: %s' % trg)
        print('Conversion: %s' % conversion)

        for roi in rois_list:
            print('ROI: %s' % roi)
            folder_path = os.path.join('output', conversion, roi)
            # first check converter is trained or not
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                if len(files) > 0:
                    print("converter has been trained")
                    continue
            else:
                print("converter is training")

            converter_training(src, trg, data_brain, rois_list, roi, vgg_dir, features_list, src_network, trg_network, src_decoder_dir, trg_decoder_dir, chunks_index_dict, opt)

if __name__ == '__main__':
    main()


