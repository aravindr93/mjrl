from torch.utils.data import Dataset
import pickle
import numpy as np

class PathDataset(Dataset):
    def __init__(self, input_files, transform=None, files=True):
        """
        Create a dataset from file paths storing actions and observations.

        """
        if files:
            self.paths = []
            for pickle_file in input_files:
                paths = pickle.load(open(pickle_file, 'rb'))
                self.paths = np.concatenate((self.paths, paths))
        else:
            self.paths = np.array(input_files)

        # transformation to be applied to the data
        self.transform = transform
        # size of action encoding
        self.action_dim = len(self.paths[0]['actions'][0])

        # TODO(Andrew): fill this
        # REMOVE?: It seems we don't use this value in run or here.
        self.reduced_obs_dim = 0

    def __len__(self):
        return len(self.paths) * len(self.paths[0]['observations'])


    def __getitem__(self, idx):
        len_single_path = len(self.paths[0]['observations'])
        idx_path = idx // len_single_path
        idx_in_path = idx % len_single_path

        # We want to concatenate 3 "frames" together to capture motion, so use
        # the current image, and the previous two images [if they exist].
        img = self.paths[idx_path]['image_pixels'][idx_in_path]
        prev_prev_img = img
        prev_img = img

        if idx_in_path > 0:
            prev_img = self.paths[idx_path]['image_pixels'][idx_in_path - 1]

        if idx_in_path > 1:
            prev_prev_img = self.paths[idx_path]['image_pixels'][idx_in_path - 2]

        img_c = np.expand_dims(img, axis=0)
        prev_img_c = np.expand_dims(prev_img, axis=0)
        prev_prev_img_c = np.expand_dims(prev_prev_img, axis=0)

        image_seq = np.concatenate((prev_prev_img_c, prev_img_c, img_c), axis=0)

        img = img.transpose((2, 0, 1))

        image_seq = image_seq.transpose((0, 3, 1, 2))

        #TODO: Make sure they are all numpy arrays (singletons are fine)
        # Create a datapoint with the observation, the action, reward, image of the current
        # state and sequence of previous two images.
        sample = {'observation': self.paths[idx_path]['observations'][idx_in_path],
                  'action': self.paths[idx_path]['actions'][idx_in_path],
                  'reward': np.array([self.paths[idx_path]['rewards'][idx_in_path]]),
                  'image': img,
                  'image_seq': image_seq}
        try:
            # TODO: make sure it's clear what this is
            # Robot info is the joint angles and other sensor data from robot
            sample['robot_info'] = self.paths[idx_path]['robot_info'][idx_in_path]
        except:
            pass

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_paths(self, sliding_window):
        return self.paths[-sliding_window:]

    def get_transform(self):
        return self.transform

    def get_action_stats(self):
        actions = np.concatenate([path["actions"] for path in self.paths])
        return dict(mean= np.mean(actions, axis=0), std=np.std(actions, axis=0))

    def get_reduced_obs_stats(self):
        actions = np.concatenate([path["reduced_obs"] for path in self.paths])
        return dict(mean= np.mean(actions, axis=0), std=np.std(actions, axis=0))

    def get_robot_info_stats(self):
        if 'robot_info' in self.paths[0]:
            robot_info = np.concatenate([path["robot_info"] for path in self.paths])
            return dict(mean= np.mean(robot_info, axis=0), std=np.std(robot_info, axis=0))
        return None

def get_dataset_from_files(files, transform=None):
    return PathDataset(files, transform)

def get_dataset_from_paths(data, transforms=None):
    return PathDataset(data, transforms, files=False)
