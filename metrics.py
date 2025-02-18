import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import cv2
import numpy as np

"""
计算一个batch中的平均指标，有宏平均和微平均两种方式；
"""


def calculate_metrics(pred: Tensor, label: Tensor, num_classes: int = 2):
    intersect_area, pred_area, label_area, pred_save = calculate_area(pred, label, num_classes)
    _, acc = accuracy(intersect_area, pred_area)
    iou_class, meaniou = mean_iou(intersect_area, pred_area, label_area)
    _, meandice = mean_dice(intersect_area, pred_area, label_area)
    kappav = kappa(intersect_area, pred_area, label_area)
    return acc, meaniou, meandice, kappav, iou_class, pred_save


def calculate_area(pred: Tensor, label: Tensor, num_classes: int = 2, threshold: float = 0.5):
    """
    计算 preds 和 labels 的各类的公共区域，以及各类的区域
    :param pred: N 1 H W
    :param label: N H W
    :param num_classes: 2
    :return:
    """
    # Convert the label to one-hot encoding
    label = F.one_hot(label, num_classes).float()  # N * H * W * C

    # Predicts are probabilities for the foreground class
    pred_binary = (pred > threshold).float()
    pred_binary = torch.cat((1 - pred_binary, pred_binary), dim=1)
    pred = F.one_hot(torch.argmax(pred_binary, dim=1), 2).float()

    pred_area = []
    label_area = []
    intersect_area = []

    for i in range(num_classes):
        pred_i = pred[:, :, :, i]
        label_i = label[:, :, :, i]
        if i == 1:
            pred_save = pred_i.numpy()[0]
            # cv2.imwrite("D:/clipseg-master/logs/rd64-uni/preout/t.png",pred_i.numpy()[0])
        pred_area_i = torch.sum(pred_i).unsqueeze(0)  # 1
        label_area_i = torch.sum(label_i).unsqueeze(0)  # 1
        intersect_area_i = torch.sum(pred_i * label_i).unsqueeze(0)  # 1
        pred_area.append(pred_area_i)
        label_area.append(label_area_i)
        intersect_area.append(intersect_area_i)

    pred_area = torch.cat(pred_area)  # num_classes
    label_area = torch.cat(label_area)  # num_classes
    intersect_area = torch.cat(intersect_area)  # num_classes

    return intersect_area, pred_area, label_area, pred_save


def mean_dice(intersect_area, pred_area, label_area):
    """
    Calculate dice.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area
    class_dice = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            dice = 0
        else:
            dice = intersect_area[i] * 2 / union[i]
        class_dice.append(dice)
    mdice = np.mean(class_dice)
    return np.array(class_dice), mdice


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def accuracy(intersect_area, pred_area):
    """
    Calculate accuracy
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), macc


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappav = (po - pe) / (1 - pe)
    return kappav


if __name__ == "__main__":
    # preds = torch.rand((1, 2, 512, 512))
    # labels = torch.argmax(torch.rand((1, 2, 512, 512)), dim=1)
    # simple test
    preds = cv2.imread("../preout/pre_6.png", cv2.IMREAD_GRAYSCALE)
    preds[preds > 1] = 1
    labels = cv2.imread("../preout/label_6.png", cv2.IMREAD_GRAYSCALE)
    labels[labels > 1] = 1
    preds = torch.tensor(preds).unsqueeze(0).long()
    preds = F.one_hot(preds, 2).float().permute(0, 3, 1, 2)
    labels = torch.tensor(labels).unsqueeze(0).long()

    accuracys, mean_ious, mean_dices, kappas = calculate_metrics(preds, labels)
    print(mean_ious)
    print(mean_dices)
    print(accuracys)
    print(kappas)
