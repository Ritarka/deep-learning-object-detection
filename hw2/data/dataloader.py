import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from pathlib import Path

class mydataloader(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        if not Path(txt_path).exists():
            raise Exception(f"{str(txt_path)} does not exist")
        
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                arr = line.split()
                
                if line[0] == "#":
                    sub_path = arr[1]
                    self.imgs_path.append(Path(txt_path).parent / "images" / sub_path)
                    # print(sub_path)
                    self.words.append([])
                else:            
                    # Extract the bounding box
                    arr = [float(num) for num in arr]
                    self.words[-1].append(arr)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            del label[6::3]
            if label[-2] == -1:
                label[-1] = -1
            else:
                label[-1] = 1
                
            label[2] += label[0]
            label[3] += label[1]
            
            annotation = np.array(label).reshape((1, 15))            
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
