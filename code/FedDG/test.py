import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from networks.unet2d import Unet2D
from utils.util import _connectivity_region_analysis
import cv2
import os


def find_contour_points(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour_points = np.vstack([c.squeeze() for c in contours])  # Changed to list comprehension
    else:
        contour_points = np.array([])  # Return empty array if no contours found
    return contour_points, contours


def _eval_dice(gt_y, pred_y, detail=False):
    class_map = {  # a map used for mapping label value to its name, used for output
        "0": "disk",
        "1": "cup"
    }
    dice = []
    for cls in range(0, 2):
        gt = gt_y[:, cls, ...]
        pred = pred_y[:, cls, ...]
        dice_this = 2 * np.sum(gt * pred) / (np.sum(gt) + np.sum(pred))
        dice.append(dice_this)
    return dice



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dict = torch.load(r'F:\PR\signal_domian\SETA\epoch_53.pth')
model = Unet2D()
model.load_state_dict(state_dict)
model = model.to(device)  # 使用统一的设备变量
model.eval()

import os

img_dir = r'C:\Users\DIY\Desktop\Fundus-doFE\Fundus\Domain2\train\ROIs\image'
save_path = './'
os.makedirs(save_path, exist_ok=True)

dice_array = []

for img_name in os.listdir(img_dir):
    image_path = os.path.join(img_dir, img_name)
    mask_path = image_path.replace('image', 'mask')

    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path))

    img = np.asarray(img, np.float32)
    mask = np.asarray(mask, np.float32)

    if mask.ndim == 3:  # expanda dimension for concatenate
        mask = np.mean(mask, axis=2)

    mask = 2 - np.array(mask / 127, dtype='uint8')
    disc = (mask > 0).astype('uint8')
    cup = (mask > 1).astype('uint8')

    img = cv2.resize(img, (384, 384))
    ori_img = img.copy()
    ori_img_mask = img.copy()

    mask = cv2.resize(mask, (384, 384))
    image = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    image = torch.from_numpy(image).float().to(device)

    disc = cv2.resize(disc, (384, 384))
    cup = cv2.resize(cup, (384, 384))

    disc = disc[..., np.newaxis]
    cup = cup[..., np.newaxis]

    data = np.concatenate((img, disc, cup), axis=2)
    mask_dice = np.expand_dims(data[..., 3:].transpose(2, 0, 1), axis=0)

    with torch.no_grad():
        pred = model(image)
        pred_y = pred.cpu().detach().numpy()

        pred_y[pred_y > 0.75] = 1
        pred_y[pred_y < 0.75] = 0

        pred_y_0 = pred_y[:, 0:1, ...]
        pred_y_1 = pred_y[:, 1:, ...]

        processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
        processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)

        processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)
        dice_subject = _eval_dice(mask_dice, processed_pred_y)
        dice_array.extend(dice_subject)

        processed_pred_y = processed_pred_y_0 + processed_pred_y_1
        results = processed_pred_y[0]
        results = results.transpose(1, 2, 0)[:, :, 0]

        cup = results.copy()
        cup_mask = mask.copy()
        cup[cup >= 1] = 255
        cup_mask[cup_mask >= 1] = 255

        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '_cup.jpg'), cup)
        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '_cup_mask.jpg'), cup_mask)

        _, u1 = find_contour_points(cup.astype(np.uint8))
        _, v1 = find_contour_points(cup_mask.astype(np.uint8))

        ori_img = cv2.drawContours(ori_img, u1, -1, (255, 255, 255), 2)
        ori_img_mask = cv2.drawContours(ori_img_mask, v1, -1, (255, 255, 255), 2)

        disc = results.copy()
        disc_mask = mask.copy()
        disc[disc == 2] = 255
        disc[disc != 255] = 0
        disc_mask[disc_mask == 2] = 255
        disc_mask[disc_mask != 255] = 0

        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '_disc.jpg'), disc)
        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '_disc_mask.jpg'), disc_mask)

        _, u1 = find_contour_points(disc.astype(np.uint8))
        _, v1 = find_contour_points(disc_mask.astype(np.uint8))

        ori_img = cv2.drawContours(ori_img, u1, -1, (0, 255, 255), 2)
        ori_img_mask = cv2.drawContours(ori_img_mask, v1, -1, (0, 255, 255), 2)

        if 0 == np.count_nonzero(disc):
            disc[0, 0] = 255

        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '_img.jpg'), ori_img)
        cv2.imwrite(os.path.join(save_path, img_name[:-4] + '_mask.jpg'), ori_img_mask)

dice_array = np.array(dice_array)
dice_avg = np.mean(dice_array, axis=0).tolist()
dice_avg2 = np.std(dice_array, axis=0).tolist()

print(dice_avg)
print(dice_avg2)
