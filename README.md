# Trimap-free background matting [WIP]

Code of MODNet is based on [official repo](https://github.com/ZHKKKe/MODNet).


## Backbones
- [MobileNetv2](https://drive.google.com/file/d/17GZLCi_FHhWo4E4wPobbLAQdBZrlqVnF/view?usp=sharing) (trained within UNet on Supervisely dataset)
- [ResNet18](https://drive.google.com/file/d/1WME_m8CCDupM6tLX6yPt-iA6gpmwQ7Sc/view?usp=sharing) (trained in DeepLabv3+ on Supervisely dataset)
- Pyramid Vision Transformer (no pretrained model yet)


## Datasets
- Supervisely - to pretrained on coarse annotations
- AISegment - to train with fine annotations
- Place365 - to augment foregrounds with different backgrounds


## Dataset formation
Trimaps are being generated on the fly during training.

To get diverse backgrounds for the same foregrounds, one can use 
COCO or Places365 dataset. Cut-and-paste approach results in unrealistic compositions. 
To do composition harmonization, one can use [Foreground-aware Semantic Representations for Image Harmonization](https://github.com/saic-vul/image_harmonization) with pre-trained models for 256x256 images or 
[Region-aware Adaptive Instance Normalization for Image Harmonization
](https://github.com/junleen/RainNet) with pre-trained models for 512x512 images.
