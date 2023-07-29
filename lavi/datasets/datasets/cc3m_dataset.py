from lavi.utils.lib import *
from lavi.datasets.datasets.vid_base_dataset import Dataset_Base


class CC3MDataset(Dataset_Base):
    def __init__(self, size_img, img_transform, size_frame, dataset, split, 
                 data_dir, part=None):
        super().__init__(size_img, img_transform, split=split,
                         size_frame=size_frame)

        self.dataset, self.part = dataset, part
        #self.size_img, self.size_frame = size_img, size_frame
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = data_dir

        txt = json.load(open(f'{data_dir}/txt_{dataset}.json', 'r'))
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

        else:
            self.lineidx = [int(p) for p in open(
                f'{self.data_dir}/{self.dataset}_train_{self.part}.lineidx'
                if self.split == 'train'
                else f'{self.data_dir}/{self.dataset}_val.lineidx', 'r')]
        
        self.read_tsv()

    def read_tsv(self):
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
            _T = self.size_frame
            _H = self.size_img
            _W = _H
            _C = 3
            img = T.zeros((_T, _C, _H, _W))


        return {'image': img, 'text_input': raw_txt}

    #def collate_batch(self, inputs):
    #    
    #    img, txt = map(list, unzip(inputs))
    #    all_imgs = T.stack(img, dim=0)
    #    all_txts = txt   
    #
    #    batch = {
    #        "image": all_imgs, "text_input": all_txts
    #        }
    #    return batch
