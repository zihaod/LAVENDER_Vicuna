import os
from lavi.datasets.datasets.base_dataset import BaseDataset


from lavi.utils.lib import *
from lavi.models.visbackbone.video_transform import (
    Normalize, Resize, CenterCrop,  ClipToTensor,
    RandomCrop, Compose)



class Dataset_Base(BaseDataset):
    def __init__(self, size_img, img_transform, split="train", size_frame=4):
        super().__init__()
        self.size_frame = size_frame
        self.split = split
        self.size_img = size_img
        self.img_transform = img_transform


    def pad_resize(self, img):
        w, h = img.size
        img = TV.transforms.Compose([
            TV.transforms.Pad([0, (w-h)//2] if w > h else [(h-w)//2, 0]),
            TV.transforms.Resize(
                [self.size_img, self.size_img]),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ])(img)
        return img

    def img_center_crop(self, img):
        img = TV.transforms.Compose([
            TV.transforms.Resize(self.size_img),
            TV.transforms.CenterCrop(
                (self.size_img, self.size_img)),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])(img)
        return img

    def vid_center_crop(self, img):
        img = Compose([
                Resize(self.size_img),
                CenterCrop(
                    (self.size_img, self.size_img)),
                ClipToTensor(channel_nb=3),
                Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])(img)
        img = img.permute(1, 0, 2, 3)
        return img

    def vid_rand_crop(self, img):
        assert self.split == "train"
        img = Compose([
            Resize(self.size_img),
            RandomCrop(
                (self.size_img, self.size_img)),
            ClipToTensor(channel_nb=3),
            Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])(img)
        # adapt from torch_videovision:
        # https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W)
        #   in the range [0, 1.0]
        # (C x T x H x W) --> (T x C x H x W)
        img = img.permute(1, 0, 2, 3)
        return img

    def img_rand_crop(self, img):
        assert self.split == "train"
        img = TV.transforms.Compose([
            TV.transforms.Resize(self.size_img),
            TV.transforms.RandomCrop(
                (self.size_img, self.size_img)),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])(img)
        return img

    def str2img(self, b):
        try:
            img = Image.fromarray(
                cv2.imdecode(
                    np.frombuffer(base64.b64decode(b), np.uint8),
                    cv2.IMREAD_COLOR)[:, :, ::-1]
                ).convert('RGB')
        except Exception:
            img = Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB')
        return img

    def sampling(self, start, end, n):
        if n == 1:
            return [int(round((start+end)/2.))]
        if n < 1:
            raise Exception("behaviour not defined for n<2")
        step = (end-start)/float(n-1)
        return [int(round(start+x*step)) for x in range(n)]

    def temporal_sample(self, list_of_b, random_sample=False):
        max_size_frame = len(list_of_b)
        if max_size_frame == 1 or self.size_frame == max_size_frame:
            return list_of_b
        if max_size_frame < self.size_frame:
            print(f"Error in size_frame",
                  f"\tasked for {size_frame} from {max_size_frame} frames")

        size_frame = min(self.size_frame, max_size_frame)
        size_clips = int(math.ceil(max_size_frame / size_frame))
        if random_sample:
            sampled_start = random.choice(range(size_clips))
            sampled_end = min(
                sampled_start + (size_frame - 1) * size_clips,
                max_size_frame - 1)
        else:
            sampled_start = 0
            sampled_end = max_size_frame - 1
        sampled_index = self.sampling(sampled_start, sampled_end, size_frame)
        sampled_video = [list_of_b[i] for i in sampled_index]
        return sampled_video

    def get_img_or_video(self, list_of_b):
        bufs = self.temporal_sample(
            list_of_b, random_sample=(self.split == 'train'))
        img = []
        for b in bufs:
            single_img = self.str2img(b)
            if self.split == "train":
                vis_transform = random.choice(self.img_transform)
                if vis_transform == "vid_rand_crop":
                    img.append(single_img)
                else:
                    if vis_transform == "pad_resize":
                        single_img = self.pad_resize(single_img)
                    elif vis_transform == "img_center_crop":
                        single_img = self.img_center_crop(single_img)
                    else:
                        single_img = self.img_rand_crop(single_img)
                    img.append(single_img.unsqueeze(0))
            else:
                if self.img_transform == ["vid_rand_crop"]:
                    vis_transform = "vid_center_crop"
                    img.append(single_img)
                else:
                    if self.img_transform == ["pad_resize"]:
                        vis_transform = "pad_resize"
                        single_img = self.pad_resize(single_img)
                    else:
                        vis_transform = "img_center_crop"
                        single_img = self.img_center_crop(single_img)
                    img.append(single_img.unsqueeze(0))

        if vis_transform == "vid_rand_crop":
            img = self.vid_rand_crop(img)
        elif vis_transform == "vid_center_crop":
            img = self.vid_center_crop(img)
        else:
            img = T.cat(img, dim=0)

        return img

