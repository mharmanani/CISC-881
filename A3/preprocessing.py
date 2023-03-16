import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tqdm

import torch
from torch import nn

import os

import SimpleITK as sitk

"""
Question 1: Re-slicing
"""

def resample_image(img_loc, reference_spacing):
    """
    Helper function to isolate the logic for resampling an image.
    
    :img_loc:
        The location of the image being resampled.
    :reference_spacing:
        The S_x_ref, S_y_ref, and S_z_ref extracted in the work above.
    """
    img = sitk.ReadImage(img_loc)
                
    dx, dy, dz = img.GetSize()
    sx, sy, sz = img.GetSpacing()
    sx_ref, sy_ref, sz_ref = reference_spacing

    sx_out, sy_out, sz_out = (1.0, 1.0, 1.0)
    reslice = lambda d,s,s_out:(np.round(d * (s/s_out)))
    dx_out, dy_out, dz_out = (reslice(dx, sx, sx_ref), 
                              reslice(dy, sy, sy_ref), 
                              reslice(dz, sz, sz_ref))

    # references: https://www.programcreek.com/python/example/123390/SimpleITK.ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_spacing)
    resampler.SetSize([int(dx_out), 
                       int(dy_out), 
                       int(dz_out)])
    resampler.SetOutputDirection(img.GetDirection())

    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(img.GetPixelIDValue())
    resampled_img = resampler.Execute(img)
    
    return resampled_img


def re_slice_cases(dir, desig, reference_spacing):
    """
    Re-slice the MRI volumes.
    
    :desig:
        A string deciding wether we are reading an "ADC", "HBV", or "T2W" image.
    
    :reference_spacing:
        The S_x_ref, S_y_ref, and S_z_ref extracted in the work above.
    """
    
    dir_=dir
    resample_dir="../DATA/resampled/cases/"
    for pid in tqdm.tqdm(os.listdir(dir_), desc="Re-slicing {0} cases".format(desig)):
        for fname in os.listdir(dir_+'/'+pid):
            if desig in fname: # check wether to open ADC/HBV/T2W
                resampled_img = resample_image(dir_+'/'+pid+'/'+fname, reference_spacing)
                resampled_img = sitk.GetArrayFromImage(resampled_img)
                np.save(resample_dir+fname, resampled_img)
    return

def re_slice_annotations(annote_type, reference_spacing, case_ids):
    """
    Re-slice the annotation files for the MRI data.
    
    :annote_type:
        A string signalling what kind of annotation (whole gland annotation or lesion)
        to resample. 
        
    :reference_spacing:
        The S_x_ref, S_y_ref, and S_z_ref extracted in the work above.
        
    :case_ids:
        A list of ids to consider when resampling. 
        This helps us avoid resampling annotations for entries not present in our data, improving performance.
    """
    
    # Choose which directory to load/save annotations and resamples
    if annote_type=='prostate':
        _dir="../DATA/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/"
        resample_dir="../DATA/resampled/annotations/whole_gland/"
    elif annote_type=='lesions':
        _dir="../DATA/picai_labels/csPCa_lesion_delineations/human_expert/"
        resample_dir="../DATA/resampled/annotations/lesions/"
    
    # Resampling logic starts here
    for fname in tqdm.tqdm(os.listdir(_dir), desc="Re-slicing {0} annotations".format(annote_type)):
        # Read .nii while avoiding archives
        if '.nii' in fname and '.gz' not in fname:
            fid = fname.split("_")[0]
            if fid not in case_ids:
                continue
            resampled_img = resample_image(_dir+fname, reference_spacing) # Resample an annotation file
            resampled_img = sitk.GetArrayFromImage(resampled_img) # Convert to numpy array
            np.save(resample_dir+fname, resampled_img) # Write the resampled annotation to disk
            
    return

"""
Question 2: Cropping
"""

def crop_volume(V, crop_size):
    """
    Crop an volume to a given size.
    
    :V:
        The volume to be cropped.
        
    :crop_size:
        The size to crop the image to.
    """
    _, x, y, z = V.shape
    a, b, c = crop_size
    return V[:, (x-a)//2:(x+a)//2, (y-b)//2:(y+b)//2, (z-c)//2:(z+c)//2]

def crop_volumes(dir, desig, subset):
    """
    Crop all volumes in a given directory to a given size.
    
    :dir:
        The directory containing the volumes to be cropped.
    """
    crop_size = (300, 300, 16)
    Vs = None
    list_of_files = [filename for filename in os.listdir(dir) if desig in filename]
    for fname in tqdm.tqdm(list_of_files, desc="Cropping {0} volumes".format(desig)):
        id = fname.split("_")[0]
        if '.npy' in fname and id in subset:
            V = np.load(dir+fname)
            V = np.transpose(V, (2, 1, 0))
            V = V.reshape(1, V.shape[0], V.shape[1], V.shape[2])
            V = crop_volume(V, crop_size)
            if Vs is None:
                Vs = V
            else:
                Vs = np.append(Vs, V, axis=0)
    return Vs

def crop_annotations(dir, subset):
    """
    Crop all annotations in a given directory to a given size.
    
    :dir:
        The directory containing the annotations to be cropped.
    """
    crop_size = (300, 300, 16)
    Vs = None
    list_of_files = os.listdir(dir)
    for fname in tqdm.tqdm(list_of_files, desc="Cropping annotations"):
        id = fname.split("_")[0]
        if '.npy' in fname and id in subset:
            V = np.load(dir+fname)
            V = np.transpose(V, (2, 1, 0))
            V = V.reshape(1, V.shape[0], V.shape[1], V.shape[2])
            V = crop_volume(V, crop_size)
            if Vs is None:
                Vs = V
            else:
                Vs = np.append(Vs, V, axis=0)
    return Vs

"""
Question 3: 2D Slicing
"""

def slice_along_z(V, slice_size=(300,300)):
    """
    Slice a volume along the z-axis.
    
    :V:
        The volume to be sliced.
        
    :slice_size:
        The size of the slices to be extracted.
    """
    x, y, z = V.shape
    a, b = slice_size
    slices = []
    for i in range(z):
        slice = torch.Tensor(V[:, :, i].astype(np.float32))
        slices.append(slice)
    return slices

"""
Question 4: Data Augmentation
"""

def flip_img_horizontal(img):
    """
    Flip an image horizontally.
    
    :V:
        The image to be flipped.
    """
    return np.flip(img, axis=1)

# Load the data
#marksheet = pd.read_csv("../DATA/picai_labels/clinical_information/marksheet.csv")

for fold in range(-1):
    re_slice_cases("../DATA/picai_public_images_fold"+str(fold)+"/", "adc", (0.5, 0.5, 3.0))
    re_slice_cases("../DATA/picai_public_images_fold"+str(fold)+"/", "hbv", (0.5, 0.5, 3.0))
    re_slice_cases("../DATA/picai_public_images_fold"+str(fold)+"/", "t2w", (0.5, 0.5, 3.0))

for fold in range(-1):
    re_slice_annotations("prostate", (0.5, 0.5, 3.0), os.listdir("../DATA/picai_public_images_fold"+str(fold)+"/"))
    re_slice_annotations("lesions", (0.5, 0.5, 3.0), os.listdir("../DATA/picai_public_images_fold"+str(fold)+"/"))


def crop_and_slice_volumes(dir, desig, subsets):
    """
    Crop and slice all volumes in a given directory.

    :dir:
        The directory containing the volumes to be cropped and sliced.

    :desig:
        The designation of the volumes to be cropped and sliced, e.g. ADC, HBV, T2W, etc.

    :subsets:
        The subset of patient IDs in the data to be cropped and sliced.
    """
    if desig in ["adc", "hbv", "t2w"]:
        volumes = crop_volumes(dir, desig, subsets)
    else:
        volumes = crop_annotations(dir, subsets)

    slices = []
    for V in volumes:
        slices += slice_along_z(V)
    return slices

def z_score_normalize_slices(slices):
    """
    Normalize the slices using z-score normalization.
    """
    for i in tqdm.tqdm(range(len(slices)), desc="Normalizing slices..."):
        slices[i] = (slices[i] - slices[i].mean()) / slices[i].std()
    return slices

def create_ids(folds):
    """
    Given a list of folds, create a list of patient IDs belonging to each 
    of the training, validation, and test sets.

    :folds:
        A list of folds, e.g. [[0, 1, 2], [3], [4]]
    """
    train_folds, validation_folds, test_folds = folds

    train_ids = []
    for fold in train_folds:
        train_ids += os.listdir("../DATA/picai_public_images_fold"+str(fold)+"/")

    validation_ids = []
    for fold in validation_folds:
        validation_ids += os.listdir("../DATA/picai_public_images_fold"+str(fold)+"/")

    test_ids = []
    for fold in test_folds:
        test_ids += os.listdir("../DATA/picai_public_images_fold"+str(fold)+"/")

    return train_ids, validation_ids, test_ids

def augment_slices(slices):
    """
    Apply data augmentation by flipping the slices
    """
    augmentations = []
    for i in tqdm.tqdm(range(len(slices)), desc="Augmenting slices..."):
        flipped_tensor = slices
        augmentations.append(flipped_tensor)
    return augmentations

def stratify(desig_samples, desig_label, subsets, no_folds=True):
    if no_folds:
        all_ids = [fname.split("_")[0] for fname in os.listdir("../DATA/resampled/annotations/lesions/")]
        
        train_idx = round(0.7 * len(all_ids))
        train_ids = all_ids[:train_idx]

        val_idx = round(0.5 * len(all_ids[train_idx:]))
        validation_ids = all_ids[train_idx:][:val_idx]
        test_ids = all_ids[train_idx:][val_idx:]
    else:
        train_ids, validation_ids, test_ids = create_ids(subsets)

    train_slices = crop_and_slice_volumes("../DATA/resampled/cases/", desig_samples, train_ids)
    train_annotations = crop_and_slice_volumes("../DATA/resampled/annotations/{0}".format(desig_label), desig_label, train_ids)

    train_data = []
    for i in range(len(train_slices)):
        train_data.append((train_slices[i], train_annotations[i]))

    del train_slices, train_annotations

    validation_slices = crop_and_slice_volumes("../DATA/resampled/cases/", desig_samples, validation_ids)
    validation_annotations = crop_and_slice_volumes("../DATA/resampled/annotations/{0}".format(desig_label), desig_label, validation_ids)

    validation_data = []
    for i in range(len(validation_slices)):
        validation_data.append((validation_slices[i], validation_annotations[i]))

    del validation_slices, validation_annotations

    test_slices = crop_and_slice_volumes("../DATA/resampled/cases/", desig_samples, test_ids)
    test_annotations = crop_and_slice_volumes("../DATA/resampled/annotations/{0}".format(desig_label), desig_label, test_ids)

    test_data = []
    for i in range(min(len(test_slices), len(test_annotations))):
        test_data.append((test_slices[i], test_annotations[i]))

    del test_slices, test_annotations

    return train_data, validation_data, test_data

def zero_padding(slice, pad_dim):
    """
    Zero-pad a slice. The padding is even in all directions.
    
    :slice:
        The slice to be padded.
        
    :pad_dim:
        The number of zeroes to add to each side of the slice.
    """
    padding = (pad_dim, ) * 4
    return torch.nn.functional.pad(slice, padding)