from utils.lib import *
from dataset import Dataset_Base, get_dl

class Dataset_Pretrain(Dataset_Base):
    def __init__(self, args, txt, dataset, split,
                 part=None, data_dir=None, tokzr=None):
        super().__init__(args, split=split,
                         size_frame=args.size_frame, tokzr=tokzr)
        if dataset in ["cc3m", "coco", "vg", "cc12m"]:
            self.size_frame = 1
        self.dataset, self.part = dataset, part
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = args.data_dir

        self.txt = txt[self.split]
        if self.dataset == "webvid10m":
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/_webvid10m-tsv_frame4/webvid10m-{self.part+1:03d}.img.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/webvid2.5m_val.lineidx', 'r')]
        elif self.dataset == "webvid10m_filtered":
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/image-1{self.part:04d}.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/webvid2.5m_val.lineidx', 'r')]
        elif self.dataset == "cc12m":
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/train.{self.part}.62.img.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/cc3m_val.lineidx', 'r')]
        else:
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/{self.dataset}_train_{self.part}.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/{self.dataset}_val.lineidx', 'r')]

    def read_tsv(self, worker_id):
        if self.dataset == "webvid10m":
            self.tsv = open(
                f'{self.data_dir}/_webvid10m-tsv_frame4/webvid10m-{self.part+1:03d}.img.tsv'
                if self.split == 'train'
                else f'{self.data_dir}/webvid2.5m_val.tsv', 'r')
        elif self.dataset == "webvid10m_filtered":
            self.tsv = open(
                f'{self.data_dir}/image-1{self.part:04d}.tsv'
                if self.split == 'train'
                else f'{self.data_dir}/webvid2.5m_val.tsv', 'r')
        elif self.dataset == "cc12m":
            self.tsv = open(
                f'{self.data_dir}/train.{self.part}.62.img.tsv'
                if self.split == 'train'
                else f'{self.data_dir}/cc3m_val.tsv', 'r')
        else:
            self.tsv = open(
                f'{self.data_dir}/{self.dataset}_train_{self.part}.tsv'
                if self.split == 'train'
                else f'{self.data_dir}/{self.dataset}_val.tsv', 'r')

    def __len__(self):
        return len(self.lineidx)

    def __getitem__(self, idx):
        lineidx = self.lineidx[idx]
        self.tsv.seek(lineidx)
        item = self.tsv.readline().split('\t')

        if self.dataset in [
                "webvid10m", "webvid10m_filtered"
                ] and self.split == "train":
            vid, bufs = item[0], item[2:]
        else:
            vid, bufs = item[0], item[1:]

        if vid in self.txt:
            raw_txt = self.txt[vid][0]
        else:
            print(f"Failed to load txt for video {vid} for ",
                  f"dataset {self.dataset}, split {self.split}, ",
                  f"part {self.part}")
            raw_txt = ""

        try:
            img = self.get_img_or_video(bufs)
            (_T, _, _H, _W) = img.shape
        except Exception as e:
            print(f"Failed to load image binaries for video {vid} for ",
                  f"dataset {self.dataset}, split {self.split}, ",
                  f"part {self.part}, {e}")
            _T = self.args.size_frame
            _H = self.args.size_img
            _W = _H
            _C = 3
            img = T.zeros((_T, _C, _H, _W))

        #txt, mask = self.str2txt(raw_txt)

        sample = {
            "image": img,
            "text_input": raw_txt,
        }

        #return img, txt, mask
        return sample

    def collate_batch(self, inputs):
        img, txt = map(list, unzip(inputs))
        all_imgs = T.stack(img, dim=0)
        all_txts = txt #T.stack(txt, dim=0)
        #all_masks = T.stack(mask, dim=0)

        batch = {
            "image": all_imgs, "text_input": all_txts
            }
        return batch
