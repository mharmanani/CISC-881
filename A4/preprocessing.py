import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

#from models import DESIClassifier, OODEstimator

def resample_image(img_loc, reference_spacing):
    """
    Helper function to isolate the logic for resampling an image.
    
    :img_loc:
        The location of the image being resampled.
    :reference_spacing:
        The S_x_ref, S_y_ref, and S_z_ref extracted in the work above.
    """
    img = sitk.ReadImage(img_loc)

    dx, dy = img.GetSize()
    print(dx, dy)
    sx, sy = img.GetSpacing()
    sx_ref, sy_ref = reference_spacing

    reslice = lambda d,s,s_out:(np.round(d * (s/s_out)))
    dx_out, dy_out = (reslice(dx, sx, sx_ref), reslice(dy, sy, sy_ref))
    print(dx_out, dy_out)

    # references: https://www.programcreek.com/python/example/123390/SimpleITK.ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_spacing)
    resampler.SetSize([int(dx_out), 
                       int(dy_out)])
    resampler.SetOutputDirection(img.GetDirection())

    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(img.GetPixelIDValue())
    resampled_img = resampler.Execute(img)
    
    return resampled_img

def scan_data_dir(data_path='../a4data/'):
    """
    Scans the data directory and returns a list of all the file names.

    :param data_path: 
        The path to the data directory.
    
    :return: A list of all the file names in the data directory.
    """
    return set([fname.split('.')[0] for fname in os.listdir(data_path)])


def build_ion_matrix(loc_str, data_path='../a4data/'):
    img_name = data_path + loc_str
    ion_data = np.load(img_name + '.npz')
    ion_matrix = ion_data['peaks']
    ion_matrix = ion_matrix.reshape(ion_data['dim_y'], ion_data['dim_x'], -1)
    return ion_matrix


def apply_TIC_normalization(data):
    """
    Applies TIC normalization to the data.

    :param data: 
        The ion matrix to normalize.
    
    :return: The normalized data.
    """
    return data / np.sum(data, axis=2, keepdims=True)

def PCA_dim_reduction(data):
    """
    Applies PCA to the data.

    :param data: 
        The ion matrix to reduce.
    
    :return: The reduced data.
    """
    pca = PCA(n_components=3)
    T = pca.fit_transform(data.reshape(-1, data.shape[2])).reshape(data.shape[0], data.shape[1], -1)
    return T

def normalize_0_1(img):
    """
    Normalizes the image to the range [0, 1].

    :param img: 
        The image to normalize.
    
    :return: 
        The normalized image.
    """
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def extract_spectra(matrix, mask):
    """
    Extracts the spectra from the image.

    :param matrix: 
        The ion matrix to extract spectra from.
    :param mask: 
        The mask to use to extract the spectra.
    
    :return: 
        The extracted spectra.
    """
    spectra = []
    labels = []

    num_to_sample = 3000
    sampled = {0:0, 1:0, 2:0}

    for m in range(mask.shape[0]):
        for n in range(mask.shape[1]):
            if sampled[mask[m, n]] < num_to_sample:
                spectra.append(matrix[m, n, :])
                labels.append(mask[m, n])
                sampled[mask[m, n]] += 1
    return np.array(spectra), np.array(labels)

def create_dataset(ion_dataset, training_keys=['rblr'], val_keys=['rmll'], test_keys=['rar']):
    """
    Creates the training, validation, and test datasets.

    :param training_keys:
        The keys to use for the training dataset.
    :param val_keys:
        The keys to use for the validation dataset.
    :param test_keys:
        The keys to use for the test dataset.

    :return:
        The training, validation, and test datasets.
    """
    training_data = []
    training_labels = []
    val_data = []
    val_labels = []
    test_data = []
    test_labels = []

    for key in training_keys:
        img = ion_dataset[key]
        mask = sitk.ReadImage('../a4data/{0}.seg.nrrd'.format(key))
        mask = sitk.GetArrayFromImage(mask)
        mask = mask.reshape(mask.shape[1], mask.shape[2])
        data, labels = extract_spectra(img, mask)
        training_data.append(data)
        training_labels.append(labels)

    for key in val_keys:
        img = ion_dataset[key]
        mask = sitk.ReadImage('../a4data/{0}.seg.nrrd'.format(key))
        mask = sitk.GetArrayFromImage(mask)
        mask = mask.reshape(mask.shape[1], mask.shape[2])
        data, labels = extract_spectra(img, mask)
        val_data.append(data)
        val_labels.append(labels)

    for key in test_keys:
        img = ion_dataset[key]
        mask = sitk.ReadImage('../a4data/{0}.seg.nrrd'.format(key))
        mask = sitk.GetArrayFromImage(mask)
        mask = mask.reshape(mask.shape[1], mask.shape[2])
        data, labels = extract_spectra(img, mask)
        test_data.append(data)
        test_labels.append(labels)

    return (
        np.concatenate(training_data),
        np.concatenate(training_labels),
        np.concatenate(val_data),
        np.concatenate(val_labels),
        np.concatenate(test_data),
        np.concatenate(test_labels),
    )