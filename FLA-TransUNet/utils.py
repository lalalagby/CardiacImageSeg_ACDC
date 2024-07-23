import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from networks.FLA_TransUNet import FocusedLinearAttention, Attention

# Define a mapping for attention types
attention_types = {
    'FocusedLinearAttention': FocusedLinearAttention,
    'Attention': Attention
}

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    # print(f'Pred sum: {pred.sum()}, GT sum: {gt.sum()}')
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(img, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    image, label = img.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)  # maintain original dimensions
    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()  # add batch and channel dimensions
    # input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()  # add batch and channel dimensions
    net.eval()
    with torch.no_grad():
        outputs = net(input)
        # print(f'Output shape: {outputs.shape}')
        prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = prediction.cpu().detach().numpy()
        # print(f'Prediction sum: {prediction.sum()}')
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = outputs

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    if test_save_path is not None:
        image = zoom(image, (x / patch_size[0], y / patch_size[1]), order=3)
        print(image.shape)
        np.savez_compressed(f"{test_save_path}/{case}.npz", image=image, label=label, prediction=prediction)

    return metric_list
