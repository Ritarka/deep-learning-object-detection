import numpy as np
import torch

from utils.box_utils import jaccard


def nms(dets, thresh):
    # print(dets)
    dets = torch.as_tensor(dets).clone()
    confs = dets[:, -1].flatten()
    xyxys = dets[:, :4]
    xyxys[:, 2:] += xyxys[:, :2]
    chosen = []

    while (confs >= 0).sum():
        best = torch.argmax(confs)
        chosen.append(best)
        # print(chosen)
        # print(xyxys)
        # print(xyxys[best])
        chosen_box = xyxys[best].unsqueeze(0)
        iou_mat = jaccard(chosen_box, xyxys)
        overlapping = torch.nonzero(iou_mat > thresh).flatten()
        xyxys[overlapping] = -1
        confs[overlapping] = -1
        # print(confs)
    # print(chosen)
    return chosen
