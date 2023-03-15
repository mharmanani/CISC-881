from preprocessing import *
from dataloaders import SegmentationDataset

from model import UNet2D

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time

torch.manual_seed(88)

# Source:
    # - https://discuss.pytorch.org/t/calculating-dice-coefficient/44154
def dice_score_coeff(pred, target):
    eps = 1e-9
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection) / (m1.sum() + m2.sum() + eps)

def prepare_input(slices, masks, pad=True):
    slices = slices.to(device).unsqueeze(1)
    masks = masks.to(device).unsqueeze(1)
    
    if pad:
        slices = zero_padding(slices, 2)
        masks = zero_padding(masks, 2)

    return slices, masks

def get_batches(data_loader, batch_size):
    num_batches = int(len(data_loader) / batch_size)
    batches = []
    for b in range(num_batches):
        batch = data_loader[b * batch_size: (b + 1) * batch_size]
        batches.append(batch.unsqueeze(0))
    return batches

def train(model, train_data, valid_data, learning_rate, batch_size, num_epochs, exp_name):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    avg_train_losses, avg_valid_losses = [], [] # for plotting
    epochs_array = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_losses, valid_losses = [], [] # for calculating average loss
        for i, batch in enumerate(train_loader):
            slices, masks = batch
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
            for i, batch in enumerate(valid_loader):
                slices, masks = batch
                slices, masks = prepare_input(slices, masks)
                outputs = model(slices)
                loss = 1 - dice_score_coeff(outputs, masks)
                valid_losses.append(loss.detach().cpu().item())
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        avg_valid_losses.append(avg_valid_loss)

        print(f"[Epoch: {epoch+1}/{num_epochs}] [Loss: {avg_train_loss: .2f}] [Validation Loss: {avg_valid_loss: .2f}] [Time: {time.time() - start_time: .2f} s]")
        torch.save(model.state_dict(), "weights/{0}_epoch_{1}.pth".format(exp_name, epoch+1))
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
    plt.savefig("{0}_loss.png".format(exp_name))


def evaluate(model, test_data, batch_size=1):
    model.eval()
    dices_coeffs = []
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    with torch.no_grad():
        for i, (slices, masks) in enumerate(test_loader):
            slices, masks = prepare_input(slices, masks)
            outputs = model(slices)
            dsc = dice_score_coeff(outputs, masks)
            dices_coeffs.append(dsc.detach().cpu().item())
    avg_dsc = sum(dices_coeffs) / len(dices_coeffs)
    print(f"[Test Dice Score Coefficient: {avg_dsc: .2f}]")

    return avg_dsc

def experiment(desig, label_type, from_epoch=None, experiment_name=''):
    assert label_type[:-1] in ["whole_gland", "lesions"]
    assert desig in ["adc", "hbv", "t2w"]

    train_data, valid_data, test_data = stratify(desig, label_type, [(1,), (3,), (0,)])

    model = UNet2D(device=device)
    
    if from_epoch:
        model.load_state_dict(torch.load("weights/{0}_epoch_{1}.pth".format(experiment_name, from_epoch)))

    train(model, train_data, valid_data, 0.01, 4, 15, experiment_name)
    
    avg_test_dsc = evaluate(model, test_data, batch_size=4)
    avg_train_dsc = evaluate(model, train_data, batch_size=4)
    avg_valid_dsc = evaluate(model, valid_data, batch_size=4)

    print(f"[Test Dice Score Coefficient: {avg_test_dsc: .2f}]")
    print(f"[Train Dice Score Coefficient: {avg_train_dsc: .2f}]")
    print(f"[Valid Dice Score Coefficient: {avg_valid_dsc: .2f}]")

def main():

    # Question 9 Experiments
    experiment("t2w", "whole_gland/", experiment_name="t2w_whole_gland_9a", from_epoch=4)
    #experiment("adc", "lesions/", experiment_name="adc_lesions_9b")
    #experiment("hbv", "lesions/", experiment_name="hbv_lesions_9c")

    

    # Question 10 Ablation Experiments
    pass # TODO: Implement ablative experiments
    


if __name__ == "__main__":
    main()