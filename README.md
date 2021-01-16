#SARAS semantic segmentation module

This module is developed under [SARAS project](https://saras-project.eu/). The semantic scene segmentation module contains multiple backbones and decoders (sematic models). The Neural Network architectures for segmentation are divided into two components: Encoder and Decoder.

- Encoder represent the pretrained backbone. This implementation contains Resnet and ResNest based backbones. 
- The dcoder represent the semantic segmentation model. We already haev implemented OCNet, DANet, UPerNet and PSPNet in the module.
- Data feeding pipelines for ADE20k and Cityscapes dataset are already available in the module.


## Requirements

* pytorch 1.5
* python 3.6 



### References
Some components of this module are takes from following packages:

- [CSAILVision/semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)
- [junfu1115/DANet](https://github.com/junfu1115/DANet)



