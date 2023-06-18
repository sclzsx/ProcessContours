import labelme
import json
import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter
from pathlib import Path


def labelme_to_mask(data_dir):
    for image_path in Path(data_dir).rglob('*.jpg'):
        image_path = str(image_path)

        json_path = image_path[:-3] + 'json'
        if not os.path.exists(json_path):
            print('Not exist:', json_path)
            continue

        image = Image.open(image_path)
        imageHeight = image.height
        imageWidth = image.width
        img_shape = (imageHeight, imageWidth)

        with open(json_path, 'r', encoding='gb18030', errors='ignore') as f:
            data = json.load(f)

        label_name_to_value = {'_background_': 0, "1": 1}
        mask, _ = labelme.utils.shape.shapes_to_label(img_shape, data['shapes'], label_name_to_value)
        # print(mask.shape, mask.dtype)

        mask = Image.fromarray(mask)
        mask_path = image_path[:-4] + '_mask.png'
        mask.save(mask_path)
    print('labelme_to_mask done.')


class SegDataset(Dataset):
    def __init__(self, dataset_dir, num_classes, input_hw, train_aug):
        self.label_paths = [str(i) for i in Path(dataset_dir).rglob('*_mask.png')]
        self.num_classes = num_classes
        self.input_hw = input_hw
        self.train_aug = train_aug

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, i):
        label_path = self.label_paths[i]
        img_path = label_path.replace('_mask.png', '.jpg')
        # print(img_path)

        image = Image.open(img_path)
        label = Image.open(label_path)

        image = image.resize(self.input_hw)
        label = label.resize(self.input_hw, Image.NEAREST)

        if self.train_aug:
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

        transform = transforms.ToTensor()

        img_tensor = transform(image)
        label_tensor = transform(label).long()

        check = 0
        if check:
            label_check = np.array(label_tensor)
            label_dict = Counter(label_check.flatten())
            label_list = [j for j in range(self.num_classes)]
            for k, v in label_dict.items():
                if k not in label_list:
                    print('error:', img_path, label_path, label_dict)
            print('label_dict:', label_dict)
            print('shape, type:', img_tensor.shape, label_tensor.shape, img_tensor.dtype, label_tensor.dtype)
            print('min, max:', torch.min(img_tensor), torch.max(img_tensor), torch.min(label_tensor),
                  torch.max(label_tensor))

        return img_tensor, label_tensor


if __name__ == '__main__':
    dataset_dir = 'datasets/beforeBZ fengqin2mm-4mpa-30mm-M001-N001'

    labelme_to_mask(dataset_dir)

    dataset = SegDataset(dataset_dir=dataset_dir, num_classes=2, input_hw=(512, 512), train_aug=True)
    for img_tensor, label_tensor in dataset:
        pass
