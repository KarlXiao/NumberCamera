from .data_loader import *
from .NumberNet import *
from .loss import *


def detection_collate(batch):
    r"""
    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    :param batch: (tuple) A tuple of tensor images and lists of annotations
    :return: A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
            3) (tensor) batch of length labels stacked on their 0 dim
    """
    targets = []
    imgs = []
    length = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        length.append(sample[2])
    return torch.stack(imgs, 0), torch.stack(targets, 0), torch.tensor(length)
