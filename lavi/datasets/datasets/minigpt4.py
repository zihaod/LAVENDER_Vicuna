from lavi.utils.lib import *
from lavi.datasets.datasets.vid_base_dataset import Dataset_Base


class MiniGPT4Dataset(Dataset_Base):
    def __init__(self, size_img, img_transform, size_frame, split, data_dir):
        super().__init__(size_img, img_transform, split=split,
                         size_frame=size_frame)

        # Initialize variables you need.


    def __len__(self):
        # return dataset length
        return 

    def __getitem__(self, idx):
        # img is a torch.tensor of shape T x C x H x W
        # text_input is the answer text (string)
        # instruction is the instruction/question text (string)

        return {'image': img, 'text_input': text_input, 'instruction_input': instruction}
