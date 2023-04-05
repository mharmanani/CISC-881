import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import build_ion_matrix, create_dataset
from models import DESIResNet


def plot_train_curves(tloss_ep, vloss_ep, tacc_ep, vacc_ep, name=''):
    plt.figure()
    plt.plot(list(range(len(tloss_ep))), tloss_ep)
    plt.plot(list(range(len(vloss_ep))), vloss_ep)
    plt.savefig("{0}_loss.png".format(name))

    plt.figure()
    plt.plot(list(range(len(tacc_ep))), tacc_ep)
    plt.plot(list(range(len(vacc_ep))), vacc_ep)
    plt.savefig("{0}_acc.png".format(name))
    

def main():
    # Key to directory mapping
    id_to_img = {
        'rar': '../a4data/2017 10 05 DESI 02 RA R',
        'rmll': '../a4data/2017 10 03 DESI 04 RML L',
        'rblr': '../a4data/2017 09 29 DESI 08 RBL R',
    }

    # Build the ion matrix for each image
    ion_dataset  = {}
    for key in id_to_img:
        name = id_to_img[key]
        ion_matrix = build_ion_matrix(name, data_path='')
        ion_dataset[key] = ion_matrix

    # Create the training and test datasets from the ion matrices
    X_train, y_train, X_val, y_val, X_test, y_test = create_dataset(ion_dataset)

    # Step 1: Train a gradient boosting classifier
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=0)
    #gb.fit(X_train, y_train)
    #print('Gradient Boosting Classifier: \n')
    #print(classification_report(y_test, gb.predict(X_test)))

    # Step 2: Train a DESI ResNet classifier
    train_data_items = []
    val_data_items = []
    test_data_items = []
    for i in range(len(X_train)):
        train_data_items.append((torch.from_numpy(X_train[i]).float(), y_train[i]))

    for i in range(len(X_val)):
        val_data_items.append((torch.from_numpy(X_val[i]).float(), y_val[i]))

    for i in range(len(X_test)):
        test_data_items.append((torch.from_numpy(X_test[i]).float(), y_test[i]))

    mask = sitk.ReadImage('../a4data/{0}.seg.nrrd'.format('rar'))
    mask = sitk.GetArrayFromImage(mask)
    mask = mask.reshape(mask.shape[1], mask.shape[2])

    plt.imsave('mask.png', mask)

    # small grid search
    idx = 0
    test_accs = {}
    test_acc_w_ood = {}
    for lr in [1e-4, 1e-5, 1e-6]:
        for wd in [1e-6]:
            for bs in [64, 128, 256]:
                exp_name = "{0}_lr{1}_wd{2}_bs{3}".format(idx, lr, wd, bs)
                print("Model #{0} ({1})".format(idx, exp_name))
                train_loader = DataLoader(train_data_items, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_data_items, batch_size=bs, shuffle=True)
                test_loader = DataLoader(test_data_items, batch_size=bs, shuffle=True)

                desi_resnet = DESIResNet()
                tloss_ep, vloss_ep, tacc_ep, vacc_ep = desi_resnet.train(train_loader, val_loader, epochs=15, 
                                                                         learning_rate=lr, weight_decay=wd)
                plot_train_curves(tloss_ep, vloss_ep, tacc_ep, vacc_ep, name=exp_name)

                segment_no_ood = desi_resnet.segment_image(ion_dataset['rar'])
                plt.imsave('_seg_no_ood{0}.png'.format(exp_name), segment_no_ood)

                segment_w_ood = desi_resnet.segment_image(ion_dataset['rar'], ood_detect=True, confidence_threshold=0.1)
                plt.imsave('_seg_ood_{0}.png'.format(exp_name), segment_w_ood)

                #desi_resnet.train_ood_estimator(ood_loader)
                #metrics = desi_resnet.test(test_loader, log_name=exp_name)
                preds_no_ood, labels_no_ood = desi_resnet.evaluate(test_loader)
                test_accuracy = accuracy_score(labels_no_ood, preds_no_ood)
                test_accs[test_accuracy] = exp_name
                print("Model {0} (No OOD) Accuacy: {1}".format(idx, test_accuracy))

                preds_ood, labels_ood = desi_resnet.evaluate(test_loader, ood_detect=True, thresh=0.1)
                test_accuracy = accuracy_score(preds_ood, labels_ood)
                print("Model {0} (OOD, t=0.25) Accuacy: {1}".format(idx, test_accuracy))   
                test_acc_w_ood[test_accuracy] = exp_name             

                idx += 1

    max_acc = max(test_accs.keys())
    print("Best model: {0}".format(test_accs[max_acc]))

    plt.figure(figsize=(20,8))
    plt.barh(y=list(test_accs.values()), width=list(test_accs.keys()))
    plt.savefig("_compare_metrics.png")

    test_accs_json = pd.Series(test_accs)
    test_accs_json.to_json("_compare_test_accs.json")
    test_accs_ood_json = pd.Series(test_acc_w_ood)
    test_accs_ood_json.to_json("_compare_test_accs_w_ood.json")

    return

if __name__ == '__main__':
    main()