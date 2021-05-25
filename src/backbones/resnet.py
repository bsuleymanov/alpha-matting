import torch
import torch.nn as nn

import torchsummary
import os, warnings, sys
#from utils import add_flops_counting_methods, flops_to_string


#------------------------------------------------------------------------------
#   BaseModel
#------------------------------------------------------------------------------
class BaseModel(nn.Module):
	def __init__(self):
		super(BaseModel, self).__init__()

	# def summary(self, input_shape, batch_size=1, device='cpu', print_flops=False):
	# 	print("[%s] Network summary..." % (self.__class__.__name__))
	# 	torchsummary.summary(self, input_size=input_shape, batch_size=batch_size, device=device)
	# 	if print_flops:
	# 		input = torch.randn([1, *input_shape], dtype=torch.float)
	# 		counter = add_flops_counting_methods(self)
	# 		counter.eval().start_flops_count()
	# 		counter(input)
	# 		print('Flops:  {}'.format(flops_to_string(counter.compute_average_flops_cost())))
	# 		print('----------------------------------------------------------------')

	def init_weights(self):
		print("[%s] Initialize weights..." % (self.__class__.__name__))
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def load_pretrained_model(self, pretrained):
		if isinstance(pretrained, str):
			print("[%s] Load pretrained model from %s" % (self.__class__.__name__, pretrained))
			pretrain_dict = torch.load(pretrained, map_location='cpu')
			if 'state_dict' in pretrain_dict:
				pretrain_dict = pretrain_dict['state_dict']
		elif isinstance(pretrained, dict):
			print("[%s] Load pretrained model" % (self.__class__.__name__))
			pretrain_dict = pretrained

		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				if state_dict[k].shape==v.shape:
					model_dict[k] = v
				else:
					print("[%s]"%(self.__class__.__name__), k, "is ignored due to not matching shape")
			else:
				print("[%s]"%(self.__class__.__name__), k, "is ignored due to not matching key")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)


#------------------------------------------------------------------------------
#   BaseBackbone
#------------------------------------------------------------------------------
class BaseBackbone(BaseModel):
	def __init__(self):
		super(BaseBackbone, self).__init__()

	def load_pretrained_model_extended(self, pretrained):
		"""
		This function is specifically designed for loading pretrain with different in_channels
		"""
		if isinstance(pretrained, str):
			print("[%s] Load pretrained model from %s" % (self.__class__.__name__, pretrained))
			pretrain_dict = torch.load(pretrained, map_location='cpu')
			if 'state_dict' in pretrain_dict:
				pretrain_dict = pretrain_dict['state_dict']
		elif isinstance(pretrained, dict):
			print("[%s] Load pretrained model" % (self.__class__.__name__))
			pretrain_dict = pretrained

		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				if state_dict[k].shape!=v.shape:
					model_dict[k] = state_dict[k]
					model_dict[k][:,:3,...] = v
				else:
					model_dict[k] = v
			else:
				print("[%s]"%(self.__class__.__name__), k, "is ignored")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)


#------------------------------------------------------------------------------
#  BaseBackboneWrapper
#------------------------------------------------------------------------------
class BaseBackboneWrapper(BaseBackbone):
	def __init__(self):
		super(BaseBackboneWrapper, self).__init__()

	def train(self, mode=True):
		if mode:
			print("[%s] Switch to train mode" % (self.__class__.__name__))
		else:
			print("[%s] Switch to eval mode" % (self.__class__.__name__))

		super(BaseBackboneWrapper, self).train(mode)
		self._freeze_stages()
		if mode and self.norm_eval:
			for module in self.modules():
				# trick: eval have effect on BatchNorm only
				if isinstance(module, nn.BatchNorm2d):
					module.eval()
				elif isinstance(module, nn.Sequential):
					for m in module:
						if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
							m.eval()

	def init_from_imagenet(self, archname):
		pass

	def _freeze_stages(self):
		pass


#------------------------------------------------------------------------------
#   Util functions
#------------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ------------------------------------------------------------------------------
#   Class of Basic block
# ------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ------------------------------------------------------------------------------
#   Class of Residual bottleneck
# ------------------------------------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ------------------------------------------------------------------------------
#   Class of ResNet
# ------------------------------------------------------------------------------
class ResNet(BaseBackbone):
    basic_inplanes = 64

    def __init__(self, in_channels, block, layers, output_stride=32, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.inplanes = self.basic_inplanes
        self.output_stride = output_stride
        self.num_classes = num_classes

        if output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(self.in_channels, self.basic_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.basic_inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, 1 * self.basic_inplanes, num_layers=layers[0], stride=strides[0],
                                       dilation=dilations[0])
        self.layer2 = self._make_layer(block, 2 * self.basic_inplanes, num_layers=layers[1], stride=strides[1],
                                       dilation=dilations[1])
        self.layer3 = self._make_layer(block, 4 * self.basic_inplanes, num_layers=layers[2], stride=strides[2],
                                       dilation=dilations[2])
        self.layer4 = self._make_layer(block, 8 * self.basic_inplanes, num_layers=layers[3], stride=strides[3],
                                       dilation=dilations[3])

        if self.num_classes is not None:
            self.fc = nn.Linear(8 * self.basic_inplanes * block.expansion, num_classes)

        self.init_weights()

    def forward(self, x):
        features = [x]
        # Stage1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        # Stage2
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        # Stage3
        x = self.layer2(x)
        features.append(x)
        # Stage4
        x = self.layer3(x)
        features.append(x)
        # Stage5
        x = self.layer4(x)
        features.append(x)
        # Classification
        if self.num_classes is not None:
            x = x.mean(dim=(2, 3))
            x = self.fc(x)
        # Output
        return features

    def _make_layer(self, block, planes, num_layers, stride=1, dilation=1, grids=None):
        # Downsampler
        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        # Multi-grids
        if dilation != 1:
            dilations = [dilation * (2 ** layer_idx) for layer_idx in range(num_layers)]
        else:
            dilations = num_layers * [dilation]
        # Construct layers
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilations[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, num_layers):
            layers.append(block(self.inplanes, planes, dilation=dilations[i]))
        return nn.Sequential(*layers)


# ------------------------------------------------------------------------------
#   Instances of ResNet
# ------------------------------------------------------------------------------
def resnet18(pretrained=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def resnet34(pretrained=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def resnet50(pretrained=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def resnet101(pretrained=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def resnet152(pretrained=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def get_resnet(num_layers, **kwargs):
    if num_layers == 18:
        return resnet18(**kwargs)
    elif num_layers == 34:
        return resnet34(**kwargs)
    elif num_layers == 50:
        return resnet50(**kwargs)
    elif num_layers == 101:
        return resnet101(**kwargs)
    elif num_layers == 152:
        return resnet152(**kwargs)
    else:
        raise NotImplementedError
