import os
import csv
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from avsbench_utils import get_v2_pallete


# categories
categories = [{'supercategory': None, 'id': 1, 'name': 'accordion', 'frequency': 'n'},
              {'supercategory': None, 'id': 2, 'name': 'airplane', 'frequency': 'f'},
              {'supercategory': None, 'id': 3, 'name': 'axe', 'frequency': 'r'},
              {'supercategory': None, 'id': 4, 'name': 'baby', 'frequency': 'f'},
              {'supercategory': None, 'id': 5, 'name': 'bassoon', 'frequency': 'n'},
              {'supercategory': None, 'id': 6, 'name': 'bell', 'frequency': 'f'},
              {'supercategory': None, 'id': 7, 'name': 'bird', 'frequency': 'f'},
              {'supercategory': None, 'id': 8, 'name': 'boat', 'frequency': 'f'},
              {'supercategory': None, 'id': 9, 'name': 'boy', 'frequency': 'f'},
              {'supercategory': None, 'id': 10, 'name': 'bus', 'frequency': 'f'},
              {'supercategory': None, 'id': 11, 'name': 'car', 'frequency': 'f'},
              {'supercategory': None, 'id': 12, 'name': 'cat', 'frequency': 'f'},
              {'supercategory': None, 'id': 13, 'name': 'cello', 'frequency': 'n'},
              {'supercategory': None, 'id': 14, 'name': 'clarinet', 'frequency': 'r'},
              {'supercategory': None, 'id': 15, 'name': 'clipper', 'frequency': 'r'},
              {'supercategory': None, 'id': 16, 'name': 'clock', 'frequency': 'f'},
              {'supercategory': None, 'id': 17, 'name': 'dog', 'frequency': 'f'},
              {'supercategory': None, 'id': 18, 'name': 'donkey', 'frequency': 'c'},
              {'supercategory': None, 'id': 19, 'name': 'drum', 'frequency': 'c'},
              {'supercategory': None, 'id': 20, 'name': 'duck', 'frequency': 'f'},
              {'supercategory': None, 'id': 21, 'name': 'elephant', 'frequency': 'f'},
              {'supercategory': None, 'id': 22, 'name': 'emergency-car', 'frequency': 'n'},
              {'supercategory': None, 'id': 23, 'name': 'erhu', 'frequency': 'n'},
              {'supercategory': None, 'id': 24, 'name': 'flute', 'frequency': 'n'},
              {'supercategory': None, 'id': 25, 'name': 'frying-food', 'frequency': 'n'},
              {'supercategory': None, 'id': 26, 'name': 'girl', 'frequency': 'f'},
              {'supercategory': None, 'id': 27, 'name': 'goose', 'frequency': 'c'},
              {'supercategory': None, 'id': 28, 'name': 'guitar', 'frequency': 'f'},
              {'supercategory': None, 'id': 29, 'name': 'gun', 'frequency': 'c'},
              {'supercategory': None, 'id': 30, 'name': 'guzheng', 'frequency': 'n'},
              {'supercategory': None, 'id': 31, 'name': 'hair-dryer', 'frequency': 'f'},
              {'supercategory': None, 'id': 32, 'name': 'handpan', 'frequency': 'n'},
              {'supercategory': None, 'id': 33, 'name': 'harmonica', 'frequency': 'n'},
              {'supercategory': None, 'id': 34, 'name': 'harp', 'frequency': 'n'},
              {'supercategory': None, 'id': 35, 'name': 'helicopter', 'frequency': 'c'},
              {'supercategory': None, 'id': 36, 'name': 'hen', 'frequency': 'c'},
              {'supercategory': None, 'id': 37, 'name': 'horse', 'frequency': 'f'},
              {'supercategory': None, 'id': 38, 'name': 'keyboard', 'frequency': 'f'},
              {'supercategory': None, 'id': 39, 'name': 'leopard', 'frequency': 'n'},
              {'supercategory': None, 'id': 40, 'name': 'lion', 'frequency': 'c'},
              {'supercategory': None, 'id': 41, 'name': 'man', 'frequency': 'f'},
              {'supercategory': None, 'id': 42, 'name': 'marimba', 'frequency': 'n'},
              {'supercategory': None, 'id': 43, 'name': 'missile-rocket', 'frequency': 'c'},
              {'supercategory': None, 'id': 44, 'name': 'motorcycle', 'frequency': 'f'},
              {'supercategory': None, 'id': 45, 'name': 'mower', 'frequency': 'r'},
              {'supercategory': None, 'id': 46, 'name': 'parrot', 'frequency': 'c'},
              {'supercategory': None, 'id': 47, 'name': 'piano', 'frequency': 'f'},
              {'supercategory': None, 'id': 48, 'name': 'pig', 'frequency': 'c'},
              {'supercategory': None, 'id': 49, 'name': 'pipa', 'frequency': 'n'},
              {'supercategory': None, 'id': 50, 'name': 'saw', 'frequency': 'r'},
              {'supercategory': None, 'id': 51, 'name': 'saxophone', 'frequency': 'r'},
              {'supercategory': None, 'id': 52, 'name': 'sheep', 'frequency': 'f'},
              {'supercategory': None, 'id': 53, 'name': 'sitar', 'frequency': 'n'},
              {'supercategory': None, 'id': 54, 'name': 'sorna', 'frequency': 'n'},
              {'supercategory': None, 'id': 55, 'name': 'squirrel', 'frequency': 'c'},
              {'supercategory': None, 'id': 56, 'name': 'tabla', 'frequency': 'n'},
              {'supercategory': None, 'id': 57, 'name': 'tank', 'frequency': 'n'},
              {'supercategory': None, 'id': 58, 'name': 'tiger', 'frequency': 'c'},
              {'supercategory': None, 'id': 59, 'name': 'tractor', 'frequency': 'c'},
              {'supercategory': None, 'id': 60, 'name': 'train', 'frequency': 'f'},
              {'supercategory': None, 'id': 61, 'name': 'trombone', 'frequency': 'n'},
              {'supercategory': None, 'id': 62, 'name': 'truck', 'frequency': 'f'},
              {'supercategory': None, 'id': 63, 'name': 'trumpet', 'frequency': 'c'},
              {'supercategory': None, 'id': 64, 'name': 'tuba', 'frequency': 'r'},
              {'supercategory': None, 'id': 65, 'name': 'ukulele', 'frequency': 'n'},
              {'supercategory': None, 'id': 66, 'name': 'utv', 'frequency': 'n'},
              {'supercategory': None, 'id': 67, 'name': 'vacuum-cleaner', 'frequency': 'c'},
              {'supercategory': None, 'id': 68, 'name': 'violin', 'frequency': 'r'},
              {'supercategory': None, 'id': 69, 'name': 'wolf', 'frequency': 'r'},
              {'supercategory': None, 'id': 70, 'name': 'woman', 'frequency': 'f'}]


def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    label = np.argmax(semantic_map, axis=-1)
    return label

def load_color_mask_in_PIL_to_Tensor(path, v_pallete, size, mode='RGB'):
    color_mask_PIL = Image.open(path).convert(mode)
    color_mask_PIL = color_mask_PIL.resize(size, Image.NEAREST)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label)  # [H, W]
    color_label = color_label.unsqueeze(0)
    return color_label  # both [1, H, W]


class AVSBenchTest(Dataset):
    def __init__(self, test_img_dir="./datasets/AVSBench-openvoc/test/JPEGImages/",
                       test_mask_dir="./datasets/AVSBench-semantic/",
                       pallete_dir="./datasets/AVSBench-semantic/label2idx.json",
                       avsbench_csv="./datasets/AVSBench-semantic/metadata.csv"):
        super(AVSBenchTest, self).__init__()

        self.test_img_dir = test_img_dir
        self.test_mask_dir = test_mask_dir
        self.pallete_dir = pallete_dir
        self.avsbench_csv = avsbench_csv

        self.frame_num = 5
        self.videos = [v for v in os.listdir(test_img_dir)]
        print("{} videos are used for test.".format(len(self.videos)))
        self.v2_pallete = get_v2_pallete(pallete_dir, num_cls=71)

    def __getitem__(self, index):
        video = self.videos[index]

        with open(self.avsbench_csv, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for info in csvreader:
                if info[1] == video:
                    video_class = info[4]
                    video_label = info[6]

        video_class_id = [0]
        for d in video_class.split("_"):
            for cate in categories:
                if d == cate["name"]:
                    video_class_id.append(cate["id"])

        # img:
        imgs_pth = []
        imgs_names = []
        img_pth = os.path.join(self.test_img_dir, video)
        for img_name in sorted(os.listdir(img_pth)):
            imgs_names.append(img_name)
            imgs_pth.append(os.path.join(img_pth, img_name))

        # gt:
        annos = []
        anno_pth = os.path.join(self.test_mask_dir, video_label, video, "labels_rgb")
        for mask_image in sorted(glob.glob(anno_pth + "/*.png")):
            color_label = np.array(load_color_mask_in_PIL_to_Tensor(mask_image, self.v2_pallete, (224, 224)))
            # filter error annotations
            color_label_mask = np.isin(color_label, video_class_id)
            color_label[~color_label_mask] = 0
            annos.append(color_label)

        # audios:
        audios = os.path.join(self.test_mask_dir, video_label, video, "audio.wav")

        return imgs_pth, annos, video, imgs_names, audios

    def __len__(self):
        return len(self.videos)


if __name__=="__main__":

    test_dataset = AVSBenchTest()

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    for n_iter, batch_data in enumerate(test_dataloader):

        imgs_pth, annos, video_name, imgs_names, audios = batch_data

        print(imgs_pth)
