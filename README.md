
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<br />

<h2 align="center">Inter-individual and inter-site neural code conversion</h2>

  <p align="center">
Haibao Wang, Jun Kai Ho, Fan L. Cheng, Shuntaro C. Aoki, Yusuke Muraki, Misato Tanaka, Yukiyasu Kamitani
<p align="center">


<div align="center">

  <a href="https://github.com/KamitaniLab/InterSiteNeuralCodeConversion/blob/main/">
    <img src="figure/NCC.png",width="800" height="300">
  </a> 

</div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/KamitaniLab/InterSiteNeuralCodeConversion.svg?style=for-the-badge
[contributors-url]: https://github.com/KamitaniLab/InterSiteNeuralCodeConversion/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/KamitaniLab/InterSiteNeuralCodeConversion.svg?style=for-the-badge
[forks-url]: https://github.com/KamitaniLab/InterSiteNeuralCodeConversion/forks
[stars-shield]: https://img.shields.io/github/issues/KamitaniLab/InterSiteNeuralCodeConversion.svg?style=for-the-badge
[stars-url]: https://github.com/KamitaniLab/InterSiteNeuralCodeConversion/stargazers
[issues-shield]: https://img.shields.io/github/stars/KamitaniLab/InterSiteNeuralCodeConversion.svg?style=for-the-badge
[issues-url]: https://github.com/KamitaniLab/InterSiteNeuralCodeConversion/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt

## Getting Started
### Installation
To begin, clone the repository on your local machine, using git clone and pasting the url of this project:
   ```sh
   git clone https://github.com/KamitaniLab/InterSiteNeuralCodeConversion.git
   ````
   
### Build Environment

Step1: Navigate to the base directory and create the Conda environment:
  ```sh
  conda env create -f env.yaml
  ```
Step2: Activate the environment:
  ```sh
  conda activate NCC
  ```
### Download Data

To use this project, you'll need to download and organize the required data:
- Download the brain data from [Figshare](https://figshare.com/articles/dataset/Inter-individual_deep_image_reconstruction/17985578).
- Download the DNN features of stimuli from [Figshare](https://figshare.com/articles/dataset/Inter-individual_and_inter-site_neural_code_conversion/26860954)

Alternatively, you can use the following commands to download specific data directly:
 ```sh
# In "data" directory:
# To download the training fMRI data:
python download.py fmri_training

# Or to download the test fMRI data:
python download.py fmri_test

# download the DNN features of training images:
python download.py stimulus_feature
 ```

### Download Pre-trained Decoders

To use this project, you'll need to download the required pre-trained decoders from [Figshare](https://figshare.com/articles/dataset/Inter-individual_and_inter-site_neural_code_conversion/26860954) with the following command:

```sh
python download.py pre-trained-decoders
```

If you prefer to train the decoders yourself (approximately 2 days per subject), detailed instructions and scripts are available in the `feature-decoding` directory.
## Usage

### Train Neural Code Converters

#### Content Loss-based Training

To train the neural code converters using content loss for subject pairs, run:

```sh
python NCC_content_loss/NCC_train.py --cuda
```

* **Note**: Use the `--cuda` flag when running on a GPU server. Omit `--cuda` if training on a CPU server.

Training one subject pair usually takes about 15 hours due to the large computational requirements. You can also download the pre-trained converters from [Figshare](https://figshare.com/articles/dataset/Inter-individual_and_inter-site_neural_code_conversion/26860954) with the following command:

```sh
python download.py pre-trained-converters
```

#### Brain Loss-based Training

To train the neural code converters using brain loss for subject pairs, run:

```sh
python NCC_brain_loss/NCC_train.py
```

### Test Neural Code Converters

#### DNN Feature Decoding

To decode DNN features from converted brain activities (approximately 80 mins per subject pair), use the following commands:

- For content loss-based converters:

  ```sh
  python NCC_content_loss/NCC_test.py
  ```

- For brain loss-based converters:

  ```sh
  python NCC_brain_loss/NCC_test.py
  ```

#### Image Reconstruction

To reconstruct images from the decoded features:

1. Navigate to the `reconstruction` directory.
2. Follow the provided README and reconstruction demo for detailed instructions on setting up the environment and usage.
3. Modify the directory of the decoded features in the script as needed to reconstruct images.

### Quantitative Evaluation
The quantitative evaluations are presented in terms of conversion accuracy, decoding accuracy, and identification accuracy.

#### Conversion Accuracy
To calculate raw correlations for conversion accuracy, navigate to the `conversion_accuracy` directory and run:

- For content loss-based converters:

  ```sh
  # pattern correlation
  python fmri_pattern_corr_content_loss.py
  
  # profile correlation
  python fmri_profile_corr_content_loss.py
  ```
  
- For brain loss-based converters:

  ```sh
  # pattern correlation
  python fmri_pattern_corr_brain_loss.py
  
  # profile correlation
  python fmri_profile_corr_brain_loss.py
  ```

To obtain the normalized correlations and plot the Figure 2E and 2F with the provided result, use the following command:
```sh
python plot_figure.py
```

#### Decoding Accuracy
To calculate decoding accuracy for decoded features, first download the ground truth features of the stimulus images using:
```sh
python download.py test_image-true_features
```

Then, navigate to the `decoding_accuracy` directory and run:
```sh
python featdec_eval.py
```
To plot the Figure 3B and 3C with the provided result, use the following command:
```sh
python plot_figure.py
```
#### Identification Accuracy
To quantitatively evaluate the reconstructed images, please request and download the ground truth stimulus images using this [link](https://forms.gle/ujvA34948Xg49jdn9) due to licensing restrictions. Organize the downloaded images in the following directory structure: `data/test_image/source`.

Then, navigate to the `identification_accuracy` directory and run:
```sh
python recon_image_eval.py
python recon_image_eval_dnn.py
```
To plot the Figure 3F with the provided result, use the following command.
```sh
python plot_figure.py
```

## Citation

Wang, H., Ho, J. K., Cheng, F. L., Aoki, S. C., Muraki, Y., Tanaka, M., & Kamitani, Y. (2024). Inter-individual and inter-site neural code conversion without shared stimuli. arXiv preprint arXiv:2403.11517.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
