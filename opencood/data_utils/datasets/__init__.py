from opencood.data_utils.datasets.v2v4real_dataset import V2V4RealDataset

__all__ = {
    'V2V4RealDataset': V2V4RealDataset,
}

# the final range for evaluation
GT_RANGE = [-80, -40, -5, 80, 40, 3]
# The communication range for cavs
COM_RANGE = 70

def build_dataset(
        dataset_cfg, 
        dataset_name=None, 
        visualize=False, 
        train=True, 
        is_self_train=False, 
        isSim=False, 
        is_check_label_quality=False, 
        data_split=None, 
        reverse_traintest=False,
        distance_filtering=False,
        min_distance=0.,
        max_distance=90.
    ):
    if dataset_name is None:
        dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        is_self_train=is_self_train,
        isSim=isSim,
        is_check_label_quality=is_check_label_quality,
        data_split=data_split,
        reverse_traintest=reverse_traintest,
        distance_filtering=distance_filtering,
        min_distance=min_distance,
        max_distance=max_distance
    )

    return dataset
