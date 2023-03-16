from preprocessing import *

from model import UNet2D

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time

import torch.nn.functional as F

import torch.nn as nn

torch.manual_seed(88)

# Sources:
    # - https://discuss.pytorch.org/t/calculating-dice-coefficient/44154
    # - https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
def dice_score_coeff(pred, target, smooth=1.):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    eps = 1e-9

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = ((2. * intersection + smooth) / (A_sum + B_sum + eps + smooth) )
    
    return dice

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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

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

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_array, avg_train_losses, label="Training Loss")
    plt.plot(epochs_array, avg_valid_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("{0}_loss.png".format(exp_name))


def evaluate(model, test_data, batch_size=1, from_epoch=None, experiment_name=''):
    model.eval()
    dices_coeffs = []
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    if from_epoch:
        model.load_state_dict(torch.load("weights/{0}_epoch_{1}.pth".format(experiment_name, from_epoch)))

    with torch.no_grad():
        for i, (slices, masks) in enumerate(test_loader):
            slices, masks = prepare_input(slices, masks)
            outputs = model(slices)
            dsc = dice_score_coeff(outputs, masks)
            dices_coeffs.append(dsc.detach().cpu().item())
    avg_dsc = sum(dices_coeffs) / len(dices_coeffs)

    return avg_dsc

def experiment(model, desig, label_type, batch_size=16, from_epoch=None, 
               experiment_name='', eval_only=False, num_epochs=10, learning_rate=1e-3):
    assert label_type[:-1] in ["whole_gland", "lesions"]
    assert desig in ["adc", "hbv", "t2w"]

    train_data, valid_data, test_data = stratify(desig, label_type, [(1,), (3,), (0,)])
    
    if from_epoch:
        model.load_state_dict(torch.load("weights/{0}_epoch_{1}.pth".format(experiment_name, from_epoch)))

    if not eval_only:
        train(model, train_data, valid_data, learning_rate, batch_size, num_epochs, experiment_name)

    avg_test_dsc = evaluate(model, test_data, batch_size=batch_size)
    avg_train_dsc = evaluate(model, train_data, batch_size=batch_size)
    avg_valid_dsc = evaluate(model, valid_data, batch_size=batch_size)

    print("{0} Results".format(experiment_name))
    print(f"[Test Dice Score Coefficient: {avg_test_dsc: .6f}]")
    print(f"[Train Dice Score Coefficient: {avg_train_dsc: .6f}]")
    print(f"[Valid Dice Score Coefficient: {avg_valid_dsc: .6f}]")

def main():
    ############################
    ## Question 9 Experiments ##
    ############################

    # T2W Whole Gland Segmentation
    model = UNet2D(device=device)
    experiment(model, "t2w", "whole_gland/", batch_size=40, experiment_name='t2w_whole_gland_9a')

    # ADC Cancer Lesions Segmentation
    model = UNet2D(device=device)
    experiment(model, "adc", "lesions/", batch_size=40, experiment_name='adc_lesions_9b')

    # HBV Cancer Lesions Segmentation
    model = UNet2D(device=device)
    experiment(model, "hbv", "lesions/", batch_size=40, experiment_name='hbv_lesions_9c')

    ###################################
    ## Question 10 Ablation Studies  ##
    ###################################

    # Ablation experiment on the learning rate
    model = UNet2D(device=device)
    experiment(model, "adc", "lesions/", batch_size=40, learning_rate=1e-4, experiment_name='adc_lesions_10_lr1')
    
    model = UNet2D(device=device)
    experiment(model, "adc", "lesions/", batch_size=40, learning_rate=1e-2, experiment_name='adc_lesions_10_lr2')
    
    model = UNet2D(device=device)
    experiment(model, "hbv", "lesions/", batch_size=40, learning_rate=1e-4, experiment_name='hbv_lesions_10_lr1')
    
    model = UNet2D(device=device)
    experiment(model, "hbv", "lesions/", batch_size=40, learning_rate=1e-2, experiment_name='hbv_lesions_10_lr2')
    
    model = UNet2D(device=device)
    experiment(model, "t2w", "whole_gland/", batch_size=40, learning_rate=1e-4, experiment_name='t2w_whole_gland_10_lr1')
    
    model = UNet2D(device=device)
    experiment(model, "t2w", "whole_gland/", batch_size=40, learning_rate=1e-2, experiment_name='t2w_whole_gland_10_lr2')

    model = UNet2D(device=device)
    experiment(model, "adc", "lesions/", batch_size=32, learning_rate=1e-3, experiment_name='adc_lesions_10_bs32')
    
    model = UNet2D(device=device)
    experiment(model, "adc", "lesions/", batch_size=16, learning_rate=1e-3, experiment_name='adc_lesions_10_bs16')
    
    model = UNet2D(device=device)
    experiment(model, "hbv", "lesions/", batch_size=32, learning_rate=1e-3, experiment_name='hbv_lesions_10_bs32')
    
    model = UNet2D(device=device)
    experiment(model, "hbv", "lesions/", batch_size=16, learning_rate=1e-3, experiment_name='hbv_lesions_10_bs16')
    
    model = UNet2D(device=device)
    experiment(model, "t2w", "whole_gland/", batch_size=32, learning_rate=1e-3, experiment_name='t2w_whole_gland_10_bs32')
    
    model = UNet2D(device=device)
    experiment(model, "t2w", "whole_gland/", batch_size=16, learning_rate=1e-3, experiment_name='t2w_whole_gland_10_bs16')




if __name__ == "__main__":
    main()