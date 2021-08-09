# Trimap-free background matting


## Results
Randomly (cherry-picked :)) picked from an experiment
- model: MODNet 
- backbone: ResNet18
- trimaps: randomly generated on each iteration
- data: AISegment, no augmentation, no image harmonization
- training: 100k iterations, 4x8 batch size
- W&B [report](https://wandb.ai/bsuleymanov/alpha-matting/reports/MODNet-ResNet18-AISegment--Vmlldzo5MTg4NjY?accessToken=xavo7dyslq9ttkt0onvfh9bf63n7zsfc2n0cn94e5akzf6zdzh5j2fckqns3azzg)

![Good result](https://github.com/bsuleymanov/alpha-matting/blob/main/images/res1.png "Result")
![Good result](https://github.com/bsuleymanov/alpha-matting/blob/main/images/res2.png "Result")
![Bad result](https://github.com/bsuleymanov/alpha-matting/blob/main/images/res3.png "Result")




## Backbones
- [MobileNetv2](https://drive.google.com/file/d/17GZLCi_FHhWo4E4wPobbLAQdBZrlqVnF/view?usp=sharing) (trained within UNet on Supervisely dataset)
- [ResNet18](https://drive.google.com/file/d/1WME_m8CCDupM6tLX6yPt-iA6gpmwQ7Sc/view?usp=sharing) (trained in DeepLabv3+ on Supervisely dataset)
- [Pyramid Vision Transformer](https://github.com/whai362/PVT) (5 pyramid stages instead of 4, no pretrained model yet)


## Datasets
- [Supervisely](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets) - to pretrained on coarse annotations
- [AISegment](https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets) - to train with fine annotations
- [Place365](https://github.com/CSAILVision/places365) - to augment foregrounds with different backgrounds


## Dataset formation
Trimaps are being generated on the fly during training.

To get diverse backgrounds for the same foregrounds, one can use 
COCO or Places365 dataset. Cut-and-paste approach results in unrealistic compositions. 
To do composition harmonization, one can use [Foreground-aware Semantic Representations for Image Harmonization](https://github.com/saic-vul/image_harmonization) with pre-trained models for 256x256 images or 
[Region-aware Adaptive Instance Normalization for Image Harmonization
](https://github.com/junleen/RainNet) with pre-trained models for 512x512 images.

## Acknowledgement
Code of MODNet architecture is based on [official repo](https://github.com/ZHKKKe/MODNet).
Code of PVT architecture is based on [official repo](https://github.com/whai362/PVT).
Einops usage is learnt from [lucidrain](https://github.com/lucidrains/vit-pytorch).
Image harmonization is based on [Foreground-aware Semantic Representations for Image Harmonization](https://github.com/saic-vul/image_harmonization) and [Region-aware Adaptive Instance Normalization for Image Harmonization
](https://github.com/junleen/RainNet).
