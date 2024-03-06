import torch
from torch import nn
import nni
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customdataset import CustomDataset
from efficientnet import EfficientNet
from utils import *

if __name__ == "__main__":

    epochs = 80
    alpha, beta = 1.2, 1.1
    batch_size = 32
    maximum_number_of_parameters = 1e6
    
    params = {
        'k_mult': 6,
        'o_c_1': 16,
        'o_c_2': 24,
        'o_c_3': 40,
        'o_c_4': 80,
        'o_c_5': 112,
        'o_c_6': 192,
        'o_c_7': 320,
        'resolution': 224,
        'dropout': 0.2,
        #'phi': 0
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

    #phi = params["phi"]
    phi = 0
    resolution = params["resolution"]
    dropout = params["dropout"]
    

    transform = transforms.Compose([
        transforms.Resize(size=resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=resolution),
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

    output_class = 10 # for imagenette
    model = EfficientNet(phi, resolution, dropout, basic_mb_params, alpha, beta, output_class)

    macs, params = get_model_params(model, resolution)

    print(f"Number of params : {params} | Resolution = {resolution}\n")

    # Si on a trop de paramètres dans le modèle, on n'effectue même pas la recherche (on renvoie une accuracy de 0)
    if params > maximum_number_of_parameters:
        nni.report_final_result(0)
        exit()
    else: 
        loss_fn = nn.CrossEntropyLoss()
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
        

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)

            accuracy = test(val_loader, model, loss_fn)
            nni.report_intermediate_result(accuracy)
        nni.report_final_result(accuracy)

        print(f"Accuracy: {accuracy} | Parameters: {params}")