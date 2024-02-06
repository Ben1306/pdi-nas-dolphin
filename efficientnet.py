import torch
from torch import nn

from math import ceil

from torchvision.ops import StochasticDepth
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customdataset import CustomDataset
from ptflops import get_model_complexity_info

import nni


if __name__ == "__main__":
    ###############################################################
    ######## EFFICIENT-NET CONVOLUTIONNAL NEURAL NETWORK
    ###############################################################

    
    # Objective function for nni optimization (not used here)
    def objective_function(accuracy, params, macs, lam_1=0.01, lam_2=0.05):
        return accuracy - lam_1 * max(0, params - 1e6) - lam_2 * max(0, macs - 500e6)


    basic_mb_params = [
        # k, out_channels(c), repeats(t), stride(s), kernel_size(k)
        [1, 16, 1, 1, 3],
        [6, 24, 2, 2, 3],
        [6, 40, 2, 2, 5],
        [6, 80, 3, 2, 3],
        [6, 112, 3, 1, 5],
        [6, 192, 4, 2, 5],
        [6, 320, 1, 1, 3],
    ]


    #####
    # This is for nni (not used here)
    #####
    """
    params = {
        'k_mult': 6,
        'o_c_1': 16,
        'o_c_2': 24,
        'o_c_3': 40,
        'o_c_4': 80,
        'o_c_5': 112,
        'o_c_6': 192,
        'o_c_7': 320,
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    

    basic_mb_params = [
        # k, out_channels(c), repeats(t), stride(s), kernel_size(k)
        [1, params["o_c_1"], 1, 1, 3],
        [params["k_mult"], params["o_c_2"], 2, 2, 3],
        [params["k_mult"], params["o_c_3"], 2, 2, 5],
        [params["k_mult"], params["o_c_4"], 3, 2, 3],
        [params["k_mult"], params["o_c_5"], 3, 1, 5],
        [params["k_mult"], params["o_c_6"], 4, 2, 5],
        [params["k_mult"], params["o_c_7"], 1, 1, 3],
    ]

    """

    alpha, beta = 1.2, 1.1

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
        def __init__(self, model_name, alpha, beta, output):
            super(EfficientNet, self).__init__()
            self.stochastic_depth_prob = 0.2
            phi, resolution, dropout = scale_values[model_name]
            self.depth_factor, self.width_factor = alpha**phi, beta**phi
            self.last_channels = ceil(1280 * self.width_factor)
            self.avgpool= nn.AdaptiveAvgPool2d(1)
            self.feature_extractor()
            self.flatten = nn.Flatten()
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(self.last_channels, output),
            )

        def feature_extractor(self):
            channels = int(32 * self.width_factor)

            features = [ConvBlock(3, channels, 3, stride=2, padding=1)]
            in_channels = channels
            
            total_stage_blocks = sum(block[2] for block in basic_mb_params)
            stage_block_id = 0
            for k, c_o, repeat, s, n in basic_mb_params:
                # For numeric stability, we multiply and divide by 4
                out_channels = 4 * ceil(int(c_o * self.width_factor) / 4)
                num_layers = ceil(repeat * self.depth_factor)

                sd_prob = self.stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                for layer in range(num_layers):
                    if layer == 0:
                        stride = s
                    else:
                        stride = 1
                    features.append(
                            MBBlock(
                                in_channels,
                                out_channels,
                                expand_ratio=k,
                                stride=stride,
                                kernel_size=n,
                                ##padding=n// 2,
                                #padding=0, 
                                stochastic_depth_prob=sd_prob
                            )
                        )
                    in_channels = out_channels

                    stage_block_id += 1

            features.append(
                ConvBlock(in_channels, self.last_channels, kernel_size=1, stride=1, padding=0)
            )
            self.extractor = nn.Sequential(*features)

        def forward(self, x):
            x = self.avgpool(self.extractor(x))
            return self.classifier(self.flatten(x))



    ###############################################################
    #################### LOAD IMAGENET DATASET
    ###############################################################
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    ### CIFAR Loader
    """
    trainset = torchvision.datasets.CIFAR10(
        root='data', 
        train=True,
        download=True,
        transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    """

    # Create custom dataset
    train_dataset = CustomDataset(root="data/imagenette/train", transform=transform)
    valid_dataset = CustomDataset(root="data/imagenette/val", transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)

    ###############################################################
    ######## TRAIN & TEST THE MODEL ON IMAGENETTE DATASET
    ###############################################################

    model_name = 'b0'
    output_class = 10 # for CIFAR10 & imagenette
    # ImageNet has 1000 classes
    model = EfficientNet(model_name, alpha, beta, output_class)

    """
    macs, params = get_model_complexity_info(
        model, 
        (3, 224, 224), 
        as_strings=True,
        print_per_layer_stat=True, 
        verbose=False
    )
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    exit()
    """
    
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        model.cuda()

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct


    # For NNI objective computation
    """
    macs, params = get_model_complexity_info(
        model, 
        (3, 224, 224), 
        as_strings=False,
        print_per_layer_stat=False, 
        verbose=False
    )
    """

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)

        accuracy = test(val_loader, model, loss_fn)
        #objective_val = objective_function(accuracy, params, macs)

        #nni.report_intermediate_result(objective_val)
    #nni.report_final_result(objective_val) 

    #print(f"Accuracy: {accuracy} | Parameters: {params} | MACS: {macs}")

    #torch.save(model.state_dict(), "efficientnet_B0_cifar10.pth")


