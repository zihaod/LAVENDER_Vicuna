from utils.lib import *
from visbackbone.video_transform import (
    Normalize, Resize, CenterCrop,  ClipToTensor,
    RandomCrop, Compose)

#from utils.tsv_file import TSVFile, CompositeTSVFile
#from utils.tsv_file_ops import tsv_reader
#from utils.load_files import (
#    load_from_yaml_file,
#    find_file_path_in_yaml, load_box_linelist_file)
from utils.logger import LOGGER
from utils.dist import get_world_size, get_rank
#from utils.data_sampler import (
#    DistributedSamplerLimited, NodeSplitSampler, IterationBasedBatchSampler)


class Dataset_Base(T.utils.data.Dataset):
    def __init__(self, args, split="train", size_frame=4, tokzr=None):
        super().__init__()
        self.args = args
        self.size_frame = size_frame
        self.split = split



    def pad_resize(self, img):
        w, h = img.size
        img = TV.transforms.Compose([
            TV.transforms.Pad([0, (w-h)//2] if w > h else [(h-w)//2, 0]),
            TV.transforms.Resize(
                [self.args.size_img, self.args.size_img]),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ])(img)
        return img

    def img_center_crop(self, img):
        img = TV.transforms.Compose([
            TV.transforms.Resize(self.args.size_img),
            TV.transforms.CenterCrop(
                (self.args.size_img, self.args.size_img)),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])(img)
        return img

    def vid_center_crop(self, img):
        img = Compose([
                Resize(self.args.size_img),
                CenterCrop(
                    (self.args.size_img, self.args.size_img)),
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
            Resize(self.args.size_img),
            RandomCrop(
                (self.args.size_img, self.args.size_img)),
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
            TV.transforms.Resize(self.args.size_img),
            TV.transforms.RandomCrop(
                (self.args.size_img, self.args.size_img)),
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
                vis_transform = random.choice(self.args.img_transform)
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
                if self.args.img_transform == ["vid_rand_crop"]:
                    vis_transform = "vid_center_crop"
                    img.append(single_img)
                else:
                    if self.args.img_transform == ["pad_resize"]:
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

    def str2txt(self, s):
        # if version.parse(transformers.__version__) >= version.parse("4.16.1"):
        #     txt = self.tokzr.encode(s)
        #     old_len = len(txt)
        #     txt = txt[:self.args.size_txt-1]
        #     new_len = len(txt)
        #     if new_len < old_len:
        #         txt = txt + [self.sep_token_id]
        #     padding_len = self.args.size_txt-len(txt)
        #     txt = txt + [self.pad_token_id]*(padding_len)
        # else:
        txt = self.tokzr.encode(
            s, padding='max_length', max_length=self.args.size_txt,
            truncation=True)
        mask = [1 if w != self.pad_token_id else 0 for w in txt]
        mask = T.LongTensor(mask)
        txt = T.LongTensor(txt)
        assert len(txt[txt == self.sep_token_id]) == 1, f'{txt}'
        return txt, mask


def get_dl(ds, args, worker_init_fn=None, collate_fn=None):
    if args.distributed:
        sp = T.utils.data.distributed.DistributedSampler(
            ds, shuffle=(ds.split == 'train'))
    else:
        if ds.split == 'train':
            sp = T.utils.data.RandomSampler(ds)
        else:
            sp = T.utils.data.SequentialSampler(ds)
    dl = T.utils.data.DataLoader(
        ds, batch_size=args.size_batch, num_workers=args.n_workers,
        pin_memory=True, sampler=sp, worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)
    return dl

def get_dl(ds, args, worker_init_fn=None, collate_fn=None):
    
    if ds.split == 'train':
        sp = T.utils.data.RandomSampler(ds)
    else:
        sp = T.utils.data.SequentialSampler(ds)
    dl = T.utils.data.DataLoader(
        ds, batch_size=args.size_batch, num_workers=args.n_workers,
        pin_memory=True, sampler=sp, worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)
    return dl


def get_tsv_dls(args, DataCls, tokzr=None):
    if tokzr is None:
        tokzr = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer)
    img_path = f'{args.data_dir}/img_{args.dataset}.tsv'
    LOGGER.info(f"rank {get_rank()}: loading video frames from {img_path}")
    lineidx_data = pickle.load(open(
        f'{args.data_dir}/img_{args.dataset}.id2lineidx.pkl', 'rb'))
    txt_path = f'{args.data_dir}/txt_{args.task}.json'
    LOGGER.info(f"rank {get_rank()}: loading text from {txt_path}")
    txt_data = json.load(open(txt_path, 'r'))
    splits = ['train', 'val']
    if 'test' in txt_data:
        splits.append('test')

    ds_all = {
        split: DataCls(
            args, img_path, txt_data, lineidx_data, split,
            tokzr=tokzr)
        for split in splits}
    log_data_len = f"data_ratio: {args.data_ratio}"
    for split in splits:
        log_data_len += f", {split}: {len(ds_all[split])}"
    LOGGER.info(log_data_len)

    dl_all = {
        split:
        get_dl(
            ds, args,
            worker_init_fn=ds.read_tsv if hasattr(ds, 'read_tsv') else None,
            collate_fn=ds.collate_batch if hasattr(ds, 'collate_batch') else None)
        for split, ds in ds_all.items()}
    dl_tr, dl_vl = [
        dl_all[split] for split in ["train", "val"]]
    dl_ts = dl_all["test"] if "test" in dl_all else None
    return dl_tr, dl_vl, dl_ts


def move_to_cuda(batch):
    if isinstance(batch, T.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


