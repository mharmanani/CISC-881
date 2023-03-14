    #validation_slices += augment_slices(validation_slices)
    #validation_slices = torch.Tensor(np.array(validation_slices, dtype=np.float32))


    
    test_slices = crop_and_slice_volumes("../DATA/resampled/cases/", desig_samples, test_ids)
    #test_slices += augment_slices(test_slices)
    #test_slices = torch.Tensor(np.array(test_slices, dtype=np.float32))

    train_annotations = crop_and_slice_volumes("../DATA/resampled/annotations/{0}".format(desig_label), desig_label, train_ids)
    #train_annotations += augment_slices(train_annotations)
    #train_annotations = torch.Tensor(np.array(train_annotations, dtype=np.float32))

    validation_annotations = crop_and_slice_volumes("../DATA/resampled/annotations/{0}".format(desig_label), desig_label, validation_ids)
    #validation_annotations += augment_slices(validation_annotations)
    #validation_annotations = torch.Tensor(np.array(validation_annotations, dtype=np.float32))

    test_annotations = crop_and_slice_volumes("../DATA/resampled/annotations/{0}".format(desig_label), desig_label, test_ids)
    #test_annotations += augment_slices(test_annotations)
    #test_annotations = torch.Tensor(np.array(test_annotations, dtype=np.float32))