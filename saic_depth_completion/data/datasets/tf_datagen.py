def tf_datagen(dataset, n_samples=None):
    for i, sample in enumerate(dataset):
        if n_samples is not None and i > n_samples:
            break
        yield [(sample[k].permute(1, 2, 0) if len(sample[k].shape) == 3 else sample[k].permute(0, 2, 3, 1)).unsqueeze(0).detach().numpy()
               for k in ['color', 'raw_depth', 'mask']]
