from efficientnet import EfficientNet
from torch import nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customdataset import CustomDataset
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os

output_path = "outputs/train_models"

alpha, beta = 1.2, 1.1
possible_k = [1, 2, 3, 4, 5, 6]
epochs = 80
maximum_params = 1.6e6

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

S_strings = [
    "[1, 8, 1, 1, 3]",
    "[k, 16, 2, 2, 3]",
    "[k, 24, 2, 2, 5]",
    "[k, 64, 3, 2, 3]",
    "[k, 88, 3, 1, 5]",
    "[k, 128, 4, 2, 5]",
    "[k, 200, 1, 1, 3]"
]

M_strings = [
    "[1, 16, 1, 1, 3]",
    "[k, 24, 2, 2, 3]",
    "[k, 32, 2, 2, 5]",
    "[k, 72, 3, 2, 3]",
    "[k, 96, 3, 1, 5]",
    "[k, 160, 4, 2, 5]",
    "[k, 280, 1, 1, 3]"
]

L_strings = [
    "[1, 16, 1, 1, 3]",
    "[k, 24, 2, 2, 3]",
    "[k, 40, 2, 2, 5]",
    "[k, 80, 3, 2, 3]",
    "[k, 112, 3, 1, 5]",
    "[k, 192, 4, 2, 5]",
    "[k, 320, 1, 1, 3]"
]

batch_size= 32

transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=224),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create custom dataset
train_dataset = CustomDataset(root="data/imagenette/train", transform=transform)
valid_dataset = CustomDataset(root="data/imagenette/val", transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

def get_params(k, size="L"):
    if size == "S":
        return [
            # k, out_channels(c), repeats(t), stride(s), kernel_size(k)
            [1, 8, 1, 1, 3],
            [k, 16, 2, 2, 3],
            [k, 24, 2, 2, 5],
            [k, 64, 3, 2, 3],
            [k, 88, 3, 1, 5],
            [k, 128, 4, 2, 5],
            [k, 200, 1, 1, 3],
        ]
    elif size == "M":
        return [
            # k, out_channels(c), repeats(t), stride(s), kernel_size(k)
            [1, 16, 1, 1, 3],
            [k, 24, 2, 2, 3],
            [k, 32, 2, 2, 5],
            [k, 72, 3, 2, 3],
            [k, 96, 3, 1, 5],
            [k, 160, 4, 2, 5],
            [k, 280, 1, 1, 3],
        ]
    elif size == "L":
        return [
            # k, out_channels(c), repeats(t), stride(s), kernel_size(k)
            [1, 16, 1, 1, 3],
            [k, 24, 2, 2, 3],
            [k, 40, 2, 2, 5],
            [k, 80, 3, 2, 3],
            [k, 112, 3, 1, 5],
            [k, 192, 4, 2, 5],
            [k, 320, 1, 1, 3],
        ]
    else:
        return False

phi = scale_values["b0"][0]
resolution = scale_values["b0"][1]
dropout = scale_values["b0"][2]
output_class = 10 # for imagenette
# ImageNet has 1000 classes

def train(device, dataloader, model, loss_fn, optimizer):
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

def test(device, dataloader, model, loss_fn):
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
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


def train_model(model, label):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        model.cuda()

    all_accuracies = []
    for t in range(epochs):
        print(f"{label} | Epoch {t+1}\n-------------------------------")
        train(device, train_loader, model, loss_fn, optimizer)

        all_accuracies.append(test(device, val_loader, model, loss_fn))

    # We take the mean of 5 last epochs as final accuracies
    if len(all_accuracies) >= 5:
        accuracy = np.mean(np.array(all_accuracies[-5:]))
    else:
        accuracy = all_accuracies[-1]
    
    return accuracy

def train_b0_models():
    # We only train models with total params < 1e6
    models_accuracy = []
    for k in possible_k:

        # Size S
        model_S = EfficientNet(phi, resolution, dropout, get_params(k, "S"), alpha, beta, output_class)
        infos_S = get_model_params(model_S, 224)
        if infos_S[1] < maximum_params:
            accuracy_S = train_model(model_S, f"Model S | k = {k}")
            models_accuracy.append((
                str(k),
                "S",
                infos_S,
                accuracy_S
            ))

        # Size M
        model_M = EfficientNet(phi, resolution, dropout, get_params(k, "M"), alpha, beta, output_class)
        infos_M = get_model_params(model_M, 224)
        if infos_M[1] < maximum_params:
            accuracy_M = train_model(model_M, f"Model M | k = {k}")
            models_accuracy.append((
                str(k),
                "M",
                infos_M,
                accuracy_M
            ))

        # Size L
        model_L = EfficientNet(phi, resolution, dropout, get_params(k, "L"), alpha, beta, output_class)
        infos_L = get_model_params(model_L, 224)
        if infos_L[1] < maximum_params:
            accuracy_L = train_model(model_L, f"Model L | k = {k}")
            models_accuracy.append((
                str(k),
                "L",
                infos_L,
                accuracy_L
            ))

    # Create intermediate directories if they do not exist
    os.makedirs(output_path, exist_ok=True)
    # Write the raw result data in a .txt file
    with open(os.path.join(output_path, "training_results.txt"), "w") as f:
        f.write("########################################################\n")
        f.write("################ Models Training Results ###############\n")
        f.write("######################################################## \n\n")

        f.write(f"We train different versions of our EfficientNet model to get their test accuracy for {epochs} epochs\n")
        f.write(f"Models are trained over Imagenette dataset\n")
        f.write(f"We limit ourselves to models which total parameters < {convert_number(maximum_params)}\n")
        f.write("We test different values of the expand ratio k\n")
        f.write("For each value of k, we compare 3 model size (depending on the number of output channels of each MBBlock layer)\n\n")

        f.write("Model size MBBlocks layer descriptions: S, M, L\n\n")
        f.write("expand ratio, out_channels, repeats, stride, kernel_size\n\n")

        f.write("| {:^20} | {:^20} | {:^20} |\n".format("Model S", "Model M", "Model L"))
        f.write("| {:^20} | {:^20} | {:^20} |\n".format("","",""))
        for (S_s,M_s,L_s) in zip(S_strings, M_strings, L_strings):
            f.write("| {:^20} | {:^20} | {:^20} |\n".format(S_s, M_s, L_s))

        f.write("\n\n\n")

        f.write("########################################################\n")
        f.write("######################## Results #######################\n")
        f.write("######################################################## \n\n")

        for model_result in models_accuracy:
            f.write("////////////////////////////////////////\n")
            f.write(f"Expand ratio = {model_result[0]} | Size = {model_result[1]}\n")
            f.write(f"Total Parameters = {convert_number(model_result[2][1])} params | Total Macs = {convert_number(model_result[2][0])} Macs\n")
            f.write(f"For {epochs} epochs, test accuracy = {convert_to_percentage(model_result[3])}\n")
            f.write("////////////////////////////////////////\n\n")

        f.write("########################################################\n")
        f.write("###################### Raw Results #####################\n")
        f.write("######################################################## \n\n")

        f.write(f"Here are the raw results of the training\n")
        f.write(f"They are of the form List[ tuple( expand_ratio, model_size, (Macs, Params), test_accuracy ) ]\n\n")

        f.write(f"{models_accuracy}")

    #################################################
    # Create some graphs to display the output data #
    #################################################
        
    ##### Macs VS accuracy
    # Create the plot
    for model_result in models_accuracy: 
        plt.scatter(model_result[2][0], model_result[3], label=f'Expand_ratio={model_result[0]} | Size={model_result[1]}')
    plt.title(f'Macs VS Test Accuracy ({epochs} epochs)')
    plt.xlabel('Number of Macs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_path, 'macs_vs_accuracy.png'))
    plt.close()
    
        
    ##### Params VS accuracy
    # Create the plot
    for model_result in models_accuracy: 
        plt.scatter(model_result[2][1], model_result[3], label=f'Expand_ratio={model_result[0]} | Size={model_result[1]}')
    plt.title(f'Parameters VS Test Accuracy ({epochs} epochs)')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy')
    plt.legend()
    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_path, 'params_vs_accuracy.png'))
    plt.close()

if __name__ == "__main__":
    train_b0_models()