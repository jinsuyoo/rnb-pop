import os
import glob
import numpy as np

# meta_info files are located at the project root (two levels above this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_META_INFO_TRAIN = os.path.join(_PROJECT_ROOT, 'meta_info', 'meta_info_V2V4Real_train.txt')
_META_INFO_TOTAL = os.path.join(_PROJECT_ROOT, 'meta_info', 'meta_info_V2V4Real_original.txt')


def get_scenario_folders(root_dir, train_split, data_split, split_val_test=False):
    assert data_split in ['train', 'val', 'test']
    if data_split == 'val':
        assert train_split == 'subset2'

    if train_split == 'subset1':
        # lines NOT starting with '#' are the training set
        with open(_META_INFO_TRAIN, 'r') as f:
            clip_paths_train = [line.rstrip() for line in f if line.rstrip()[0] != '#']
    elif train_split == 'subset2':
        # lines starting with '#' are the training set (strip the '#')
        with open(_META_INFO_TRAIN, 'r') as f:
            clip_paths_train = [line.rstrip()[1:] for line in f if line.rstrip()[0] == '#']
    else:
        raise ValueError(f"Unknown train_split: {train_split}")

    with open(_META_INFO_TOTAL, 'r') as f:
        clip_paths_total = [line.rstrip() for line in f]

    clip_paths_test = [p for p in clip_paths_total if p not in clip_paths_train]

    if data_split == 'train':
        scenario_folders = [os.path.join(root_dir, p) for p in clip_paths_train]
    else:
        scenario_folders = [os.path.join(root_dir, p) for p in clip_paths_test]
        if train_split == 'subset2' and split_val_test:
            val_indices = [2, 7, 12, 17, 22, 27, 32]
            if data_split == 'val':
                scenario_folders = np.take(scenario_folders, val_indices).tolist()
            else:
                scenario_folders = np.delete(scenario_folders, val_indices).tolist()

    scenario_folders = sorted(scenario_folders)

    num_frames_per_clip = []
    for clip_path in scenario_folders:
        agent_id = None
        for folder in sorted(os.listdir(clip_path)):
            if folder.isdigit():
                agent_id = folder
                break
        if agent_id is None:
            raise ValueError(f'Agent id not found in {clip_path}')
        num_frames_per_clip.append(len(glob.glob(os.path.join(clip_path, agent_id, '*.pcd'))))

    print(f'data split: {data_split} / train split: {train_split} / # clips: {len(scenario_folders)} / total # frames: {sum(num_frames_per_clip)}')

    return scenario_folders
