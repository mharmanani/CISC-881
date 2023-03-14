from preprocessing import *
from dataloaders import SegmentationDataset

from model import UNet2D

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time

def dice_score_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def prepare_input(slices, masks, pad=True):
    slices = slices.to(device).unsqueeze(0).unsqueeze(0)
    masks = masks.to(device).unsqueeze(0).unsqueeze(0)
    
    if pad:
        slices = zero_padding(slices, 2)
        masks = zero_padding(masks, 2)

    return slices, masks

def train(model, train_loader, valid_loader, optimizer, num_epochs, device):
    avg_train_losses, avg_valid_losses = [], [] # for plotting
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_losses, valid_losses = [], [] # for calculating average loss
        epochs_array = []
        for i, (slices, masks) in enumerate(train_loader):
            slices, masks = prepare_input(slices, masks)
            optimizer.zero_grad()
            outputs = model(slices)
            loss = 1 - dice_score_coeff(outputs, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            for i, (slices, masks) in enumerate(valid_loader):
                slices, masks = prepare_input(slices, masks)
                outputs = model(slices)
                loss = 1 - dice_score_coeff(outputs, masks)
                valid_losses.append(loss.detach().cpu().item())
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        avg_valid_losses.append(avg_valid_loss)

        print(f"[Epoch: {epoch+1}/{num_epochs}] [Loss: {avg_train_loss: .2f}] [Validation Loss: {avg_valid_loss: .2f}] [Time: {time.time() - start_time: .2f} s]")

        epochs_array.append(epoch+1)

    print(epochs_array)
    print(avg_train_losses)
    print(avg_valid_losses)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_array, avg_train_losses, label="Training Loss")
    plt.plot(epochs_array, avg_valid_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")


def test(model, test_loader, criterion, num_epochs, device):
    model.eval()
    with torch.no_grad():
        for i, (slices, masks) in enumerate(test_loader):
            slices = slices.to(device)
            masks = masks.to(device)
            outputs = model(slices)
            loss = criterion(outputs, masks)
            print(f"Step: {i+1}/{len(test_loader)}, Loss: {loss.item():.4f}")

def experiment(desig, label_type, experiment_name=''):
    assert label_type in ["whole_gland", "lesions"]
    assert desig in ["adc", "hbv", "t2w"]

    train_data, valid_data, test_data = stratify(desig, label_type, [(1,), (3,), (0,)])
    model = UNet2D(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, train_data, valid_data, optimizer, 2, device)
    test(model, test_data, 1, 2, device)

def main():
    experiment("t2w", "whole_gland", "t2w_whole_gland_9a")
    experiment("adc", "lesions", "adc_lesions_9b")
    experiment("hbv", "lesions", "hbv_lesions_9c")
    


if __name__ == "__main__":
    main()