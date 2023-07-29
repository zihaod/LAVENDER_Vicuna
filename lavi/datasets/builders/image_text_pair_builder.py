import os
import logging
import warnings
import torch

from lavi.common.registry import registry
from lavi.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavi.datasets.datasets.webvid_dataset import WebVidDataset
from lavi.datasets.datasets.cc3m_dataset import CC3MDataset


@registry.register_builder("webvid")
class WebVidBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebVidDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/filtered.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        if isinstance(self.config.part, int):
            datasets['train'] = dataset_cls(
                size_img=self.config.size_img, 
                img_transform=self.config.img_transform,
                size_frame=self.config.size_frame, 
                #txt=self.config.txt, 
                dataset=self.config.dataset, 
                split=self.config.split, 
                data_dir=self.config.data_dir, 
                part=self.config.part
            )
        else: #part should be int list
            all_datasets = []
            for i in self.config.part:
                all_datasets.append(
                    dataset_cls(
                        size_img=self.config.size_img, 
                        img_transform=self.config.img_transform,
                        size_frame=self.config.size_frame, 
                        #txt=self.config.txt, 
                        dataset=self.config.dataset, 
                        split=self.config.split, 
                        data_dir=self.config.data_dir, 
                        part=i
                    )
                )
            datasets['train'] = torch.utils.data.ConcatDataset(all_datasets)
  
        return datasets
        

@registry.register_builder("cc3m")
class CC3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = CC3MDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc3m/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        #self.build_processors()

        #build_info = self.config.build_info
        #storage_path = build_info.storage

        datasets = dict()

        #if not os.path.exists(storage_path):
        #    warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        if isinstance(self.config.part, int):
            datasets['train'] = dataset_cls(
                size_img=self.config.size_img, 
                img_transform=self.config.img_transform,
                size_frame=self.config.size_frame, 
                #txt=self.config.txt, 
                dataset=self.config.dataset, 
                split=self.config.split, 
                data_dir=self.config.data_dir, 
                part=self.config.part
            )
        else: #part should be int list
            all_datasets = []
            for i in self.config.part:
                all_datasets.append(
                    dataset_cls(
                        size_img=self.config.size_img, 
                        img_transform=self.config.img_transform,
                        size_frame=self.config.size_frame, 
                        #txt=self.config.txt, 
                        dataset=self.config.dataset, 
                        split=self.config.split, 
                        data_dir=self.config.data_dir, 
                        part=i
                    )
                )
            datasets['train'] = torch.utils.data.ConcatDataset(all_datasets)
  
        return datasets
