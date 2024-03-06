from torch import nn
from math import ceil
from torchvision.ops import StochasticDepth


## This is a reminder of the values of these parameters for the different versions of EfficientNet
scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 288, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

# This is a helper function for MBConv Block
def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnnblock(x)
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class MBBlock(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels,
                kernel_size,
                stride, 
                #padding, 
                expand_ratio, 
                stochastic_depth_prob, 
                reduction=2
    ):
        super(MBBlock, self).__init__()
        
        
        self.use_residual = in_channels == out_channels and stride == 1

        # MODIF
        # hidden_dim = in_channels * expand_ratio
        hidden_dim = make_divisible(in_channels  * expand_ratio, 8)

        self.expand = in_channels != hidden_dim

        # This is for squeeze and excitation block
        # MODIF
        # reduced_dim = int(in_channels / reduction)
        reduced_dim = max(1, in_channels // 4)


        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim,
                kernel_size=1,stride=1,padding=0)

        self.conv = nn.Sequential(
                ConvBlock(hidden_dim,hidden_dim,kernel_size,
                stride,(kernel_size-1)//2,groups=hidden_dim),
                SqueezeExcitation(hidden_dim, reduced_dim),
                nn.Conv2d(hidden_dim, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, inputs):
        if self.expand:
            x = self.expand_conv(inputs)
        else:
            x = inputs
        x = self.conv(x)
        if self.use_residual:
            x = self.stochastic_depth(x)
            x += inputs
        return x
    

class EfficientNet(nn.Module):
    def __init__(self, phi, resolution, dropout, basic_mb_params, alpha, beta, output):
        super(EfficientNet, self).__init__()
        self.stochastic_depth_prob = 0.2
        self.phi = phi
        self.resolution = resolution
        self.dropout =dropout
        self.basic_mb_params = basic_mb_params
        self.depth_factor, self.width_factor = alpha**self.phi, beta**self.phi
        self.last_channels = ceil(1280 * self.width_factor)
        self.avgpool= nn.AdaptiveAvgPool2d(1)
        self.feature_extractor(self.resolution)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=True),
            nn.Linear(self.last_channels, output),
        )

    def feature_extractor(self, resolution):
        channels = int(32 * self.width_factor)

        features = [ConvBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        # MBBlock params number and feature map sizes
        mb_block_parameters = []
        mb_block_input_sizes = []
        mb_block_output_sizes = []

        
        total_stage_blocks = sum(block[2] for block in self.basic_mb_params)
        stage_block_id = 0
        for k, c_o, repeat, s, n in self.basic_mb_params:
            # For numeric stability, we multiply and divide by 4
            out_channels = 4 * ceil(int(c_o * self.width_factor) / 4)
            num_layers = ceil(repeat * self.depth_factor)

            sd_prob = self.stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

            for layer in range(num_layers):
                if layer == 0:
                    stride = s
                else:
                    stride = 1
                mb_block = MBBlock(
                    in_channels,
                    out_channels,
                    expand_ratio=k,
                    stride=stride,
                    kernel_size=n,
                    ##padding=n// 2,
                    #padding=0, 
                    stochastic_depth_prob=sd_prob
                )
                features.append(mb_block)


                # Track input feature map size
                input_h = ceil(mb_block_input_sizes[-1][1] / stride) if mb_block_input_sizes else ceil(resolution / (2 ** stage_block_id))
                input_w = ceil(mb_block_input_sizes[-1][2] / stride) if mb_block_input_sizes else ceil(resolution / (2 ** stage_block_id)) 
                input_size = (in_channels, input_h, input_w)
                mb_block_input_sizes.append(input_size)

                # Calculate output feature map size
                out_h = ceil(input_size[1] / stride)
                out_w = ceil(input_size[2] / stride)
                output_size = (out_channels, out_h, out_w)
                mb_block_output_sizes.append(output_size)


                in_channels = out_channels

                ## Get the number of params of the MBBlock
                num_params = sum(p.numel() for p in mb_block.parameters() if p.requires_grad)
                mb_block_parameters.append(num_params)


                stage_block_id += 1

        features.append(
            ConvBlock(in_channels, self.last_channels, kernel_size=1, stride=1, padding=0)
        )
        self.extractor = nn.Sequential(*features)

        self.mb_block_parameters = mb_block_parameters
        self.mb_block_input_sizes = mb_block_input_sizes
        self.mb_block_output_sizes = mb_block_output_sizes

    def forward(self, x):
        x = self.avgpool(self.extractor(x))
        return self.classifier(self.flatten(x))