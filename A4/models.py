import time
import torch
from torch import nn

from torchvision.models import resnet18, resnet101

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, balanced_accuracy_score as accuracy_score

class OODEstimator:
    def __init__(self, k=10):
        self.knn = NearestNeighbors(n_neighbors=k)

    def fit(self, X, y):
        self.knn.fit(X, y)

    def ood_scores(self, X):
        dist, _ = self.knn.kneighbors(X)
        return np.mean(dist, axis=1)

class DESIResNet(nn.Module):
    def __init__(self, device='cuda'):
        super(DESIResNet, self).__init__()
        self.model = nn.Sequential(
            resnet18(),
            nn.Linear(1000, 3),
            nn.Softmax(dim=-1)
        )
        self.sigmoid = nn.Sigmoid()

        self.ood_estim = OODEstimator()
        self.device = device
        self.model = self.model.to(self.device)

        if torch.cuda.device_count() > 1:
            print("using", torch.cuda.device_count(), "GPUs...")
            self.model = nn.DataParallel(self.model)
        

    def __call__(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        resnet = list([*self.model.children()])[0]
        feat_extractor = nn.Sequential(*list([resnet.children()])[:-1])

        return feat_extractor(x)

    def train_ood_estimator(self, train_loader):
        embs = []
        labels = []
        for spect, label in train_loader:
            emb = self.get_features(spect)
            embs.append(emb.detach().cpu())
            labels.append(label.detach().cpu())
        
        embs = torch.concat(embs, dim=0).numpy()
        labels = torch.concat(labels, dim=0).numpy()

        self.ood_estim.fit(embs, labels)

    def train(self, train_loader, val_loader, epochs, learning_rate=1e-3, weight_decay=1e-6):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        preds = []
        trues = []
        losses = []
        preds_val = []
        trues_val = []
        val_losses = []

        train_acc_per_epoch = []
        val_acc_per_epoch = []
        train_loss_per_epoch = []
        val_loss_per_epoch = []

        for epoch in range(epochs):
            for spect, label in train_loader:
                spect = spect.unsqueeze(1).to(self.device)
                spect = spect.repeat(1,3,1).float()
                spect = spect.unsqueeze(2)
                label = label.to(self.device)
                pred = self.model(spect)
                optimizer.zero_grad()
                loss = criterion(pred.squeeze(), label.long())
                loss.backward()
                optimizer.step()

                pred = torch.argmax(pred, dim=-1)
                preds += [yp for yp in pred.detach().cpu().numpy()]
                trues += [yt for yt in label.detach().cpu().numpy()]
                losses.append(loss.detach().cpu().numpy())

            for spect, label in val_loader:
                spect = spect.unsqueeze(1).to(self.device)
                spect = spect.repeat(1,3,1).float()
                spect = spect.unsqueeze(2)
                label = label.to(self.device)
                pred = self.model(spect)
                val_loss = criterion(pred.squeeze(), label.long())

                pred = torch.argmax(pred, dim=-1)
                preds_val += [yp for yp in pred.detach().cpu().numpy()]
                trues_val += [yt for yt in label.detach().cpu().numpy()]
                val_losses.append(val_loss.detach().cpu().numpy())

            msg_template = "[Epoch {0}] Loss={1} Val. Loss={2} Train Acc.={3} Val. Acc.={4}"
            
            train_acc_per_epoch.append(accuracy_score(trues, preds))
            val_acc_per_epoch.append(accuracy_score(trues_val, preds_val))
            train_loss_per_epoch.append(np.mean(losses))
            val_loss_per_epoch.append(np.mean(val_losses))

            print(msg_template.format(
                epoch, np.mean(losses), np.mean(val_losses), 
                accuracy_score(trues, preds),
                accuracy_score(trues_val, preds_val)
            ))
        return train_loss_per_epoch, val_loss_per_epoch, train_acc_per_epoch, val_acc_per_epoch

    def evaluate(self, test_loader, ood_detect=False, thresh=1.0):
        preds = []
        labels_filtered = []
        for spect, label in test_loader:
            spect = spect.unsqueeze(1).to(self.device)
            spect = spect.repeat(1,3,1).float()
            spect = spect.unsqueeze(2)
            label = label.to(self.device)
            probs = self.model(spect)
            uncertainty_scores = (1.0 - probs.max(-1).values) 
            uncertainty_mask = uncertainty_scores < thresh
            print(uncertainty_scores)
            if ood_detect:
                pred = torch.argmax(probs, dim=-1) * uncertainty_mask
            else:
                pred = torch.argmax(probs, dim=-1)
            
            preds += (pred.detach().cpu().numpy()).tolist()
            labels_filtered += (label.detach().cpu().numpy()).tolist()
        
        return preds, labels_filtered
    
    def segment_image(self, matrix, ood_detect=False, confidence_threshold=0.5):
        spectra = []
        out_mask = np.zeros(matrix.shape[:-1])

        for m in range(matrix.shape[0]):
            spectra = torch.from_numpy(matrix[m, :])
            spectra = spectra.unsqueeze(1).unsqueeze(1).to(self.device)
            spectra = spectra.repeat(1,3,1,1).float()
            probs = self.model(spectra)
            uncertainty_scores = (1.0 - probs.max(-1).values)
            uncertainty_mask = uncertainty_scores < confidence_threshold
            if ood_detect:
                pred = torch.argmax(probs, dim=-1) * uncertainty_mask
            else:
                pred = torch.argmax(probs, dim=-1)
            out_mask[m, :] = pred.detach().cpu().numpy()
        return out_mask