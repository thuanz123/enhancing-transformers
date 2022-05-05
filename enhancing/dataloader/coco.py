# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import json
import albumentations as A
from omegaconf import OmegaConf
from typing import Optional, List, Callable, Union, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..utils.general import initialize_from_config


class COCOBase(Dataset):
    def __init__(self, dataroot: str = "", labelroot: str = "", stuffthingroot: str = "", split: str = "",
                 onehot_segmentation: bool = False, use_stuffthing: bool = False,
                 tokenizer: Optional[OmegaConf] = None, transform: Optional[Callable] = None) -> None:
        assert split in ["train", "val"]
        self.split = split
        
        self.onehot = onehot_segmentation       # return segmentation as rgb or one hot
        self.stuffthing = use_stuffthing        # include thing in segmentation
        if self.onehot and not self.stuffthing:
            raise NotImplemented("One hot mode is only supported for the "
                                 "stuffthings version because labels are stored "
                                 "a bit different.")

        data_json = Path(labelroot)/f"captions_{split}2017.json"
        with open(data_json) as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_captions = dict()
            self.img_id_to_filepath = dict()
            self.img_id_to_segmentation_filepath = dict()
            
        if self.stuffthing:
            self.segmentation_prefix = Path(stuffthingroot)/f"{split}2017"
        else:
            self.segmentation_prefix = Path(labelroot)/f"stuff_{split}2017_pixelmaps"

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in imagedirs:
            self.img_id_to_filepath[imgdir["id"]] = Path(dataroot)/f"{split}2017"/imgdir["file_name"]
            self.img_id_to_captions[imgdir["id"]] = list()
            pngfilename = imgdir["file_name"].replace("jpg", "png")
            self.img_id_to_segmentation_filepath[imgdir["id"]] = Path(self.segmentation_prefix)/pngfilename
            self.labels["image_ids"].append(imgdir["id"])

        capdirs = self.json_data["annotations"]
        for capdir in capdirs:
            # there are in average 5 captions per image
            self.img_id_to_captions[capdir["image_id"]].append(np.array([capdir["caption"]]))

        self.transform = transform
        self.tokenizer = initialize_from_config(tokenizer)

    def __len__(self):
        return len(self.labels["image_ids"])

    def preprocess_image(self, image_path, segmentation_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        segmentation = Image.open(segmentation_path)
        if not self.onehot and not segmentation.mode == "RGB":
            segmentation = segmentation.convert("RGB")
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.onehot:
            assert self.stuffthing
            # stored in caffe format: unlabeled==255. stuff and thing from
            # 0-181. to be compatible with the labels in
            # https://github.com/nightrome/cocostuff/blob/master/labels.txt
            # we shift stuffthing one to the right and put unlabeled in zero
            # as long as segmentation is uint8 shifting to right handles the
            # latter too
            assert segmentation.dtype == np.uint8
            segmentation = segmentation + 1

        image, segmentation = self.transform(image=image, segmentation=segmentation)
        image = (image / 255).astype(np.float32)

        if self.onehot:
            assert segmentation.dtype == np.uint8
            # make it one hot
            n_labels = 183
            flatseg = np.ravel(segmentation)
            onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)
            onehot[np.arange(flatseg.size), flatseg] = True
            onehot = onehot.reshape(segmentation.shape + (n_labels,)).astype(int)
            segmentation = onehot
        else:
            segmentation = (segmentation / 255).astype(np.float32)
            
        return image, segmentation

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
        image, segmentation = self.preprocess_image(img_path, seg_path)
        
        captions = self.img_id_to_captions[self.labels["image_ids"][i]]
        caption = captions[np.random.randint(0, len(captions))]
        caption = self.tokenizer.tokenize(caption).squeeze(0)
        
        return {"image": image, "caption": caption, "segmentation": segmentation}


class COCOTrain(COCOBase):
    def __init__(self, dataroot: str, labelroot: str, stuffthingroot: str, tokenizer: OmegaConf,
                 resolution: Union[Tuple[int, int], int], onehot_segmentation: bool = False, use_stuffthing: bool = False) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = A.Compose(
            [A.SmallestMaxSize(max_size=min(resolution)),
             A.RandomCrop(height=resolution[0], width=resolution[1])],
            additional_targets={"segmentation": "image"})
        
        super().__init__(dataroot, labelroot, stuffthingroot, "train",
                         onehot_segmentation, use_stuffthing, tokenizer, transform)


class COCOValidation(COCOBase):
    def __init__(self, dataroot: str, labelroot: str, stuffthingroot: str, tokenizer: OmegaConf,
                 resolution: Union[Tuple[int, int], int], onehot_segmentation: bool = False, use_stuffthing: bool = False) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = A.Compose(
            [A.SmallestMaxSize(max_size=min(resolution)),
             A.CenterCrop(height=resolution[0], width=resolution[1])],
            additional_targets={"segmentation": "image"})
        
        super().__init__(dataroot, labelroot, stuffthingroot, "val",
                         onehot_segmentation, use_stuffthing, tokenizer, transform)
