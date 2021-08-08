from .wrapper import *


#------------------------------------------------------------------------------
#  Replaceable Backbones
#------------------------------------------------------------------------------

SUPPORTED_BACKBONES = {
    'mobilenetv2': MobileNetV2Backbone,
    'resnet18': ResNet18Backbone,
    'pvt': PVTBackbone
}
