import torch

from dataset import IGNORE_INDEX

NUM_CLASSES = 3


def update_confusion(conf, pred, target):
    """conf[true, predicted] += 1 for every valid pixel."""
    valid = target != IGNORE_INDEX
    pred, target = pred[valid], target[valid]
    idx = target * NUM_CLASSES + pred
    conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_confusion(conf):
    conf = conf.float()
    tp = conf.diag()
    union = conf.sum(1) + conf.sum(0) - tp
    return tp / union.clamp(min=1)


def precision_recall_from_confusion(conf):
    conf = conf.float()
    tp = conf.diag()
    precision = tp / conf.sum(0).clamp(min=1)   
    recall = tp / conf.sum(1).clamp(min=1)      
    return precision, recall