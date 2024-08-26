# DNN feature decoding

The scripts of deep neural network (DNN) feature decoding from fMRI brain activities is originally proposed by [Horikawa & Kamitani (2017)](https://www.nature.com/articles/ncomms15037) and employed in DNN-based image reconstruction methods of [Shen et al. (2019)](http://dx.doi.org/10.1371/journal.pcbi.1006633) as well as recent studies in Kamitani lab.

## Usage

### Decoding with PyFastL2LiR

Example config file: [deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml](config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml)

- Training: `python featdec_fastl2lir_train.py deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml`
- Test (prediction): `python featdec_fastl2lir_predict.py deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml`


## References

- Horikawa and Kamitani (2017) Generic decoding of seen and imagined objects using hierarchical visual features. *Nature Communications* 8:15037. https://www.nature.com/articles/ncomms15037
- Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity. *PLOS Computational Biology*. https://doi.org/10.1371/journal.pcbi.1006633
