from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)
import ipdb
import sys
from copy import copy, deepcopy
import h5py, os
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import random
sys.path.append("..") 

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0
        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        return self._transform(post_index)
class LoadImaged_BodyMap(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)



    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)

        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):

            # ==================== 💡 1. 改进错误处理 ====================
            try:
                # ==================== 💡 2. 修复变量名混用 ====================
                # 使用一个新变量名 `loaded_item`，避免覆盖输入的 `data` 字典
                loaded_item = self._loader(d[key], reader)

            except Exception as e:
                # 打印详细的错误信息，帮助你找到根本原因（比如文件路径问题）
                print(f"--- ERROR: Failed to load key '{key}' for sample '{d.get('name', 'N/A')}' ---")
                print(f"File path was: {d.get(key, 'N/A')}")
                print(f"Exception: {e}")
                # 安全地返回原始数据字典，让 CacheDataset 跳过这个坏样本
                return d

            if self._loader.image_only:
                d[key] = loaded_item
            else:
                # 这里不再需要检查 loaded_item 的类型，因为如果加载失败，上面已经返回了
                d[key] = loaded_item[0]
                if not isinstance(loaded_item[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = loaded_item[1]

        # 现在 d['image'] 已经是加载好的图像数组了，这行代码可以正常执行
        d['label'], d['label_meta_dict'] = self.label_transfer(d['label'], d['image'].shape)

        return d

    def label_transfer(self, lbl_dir, shape):
        organ_lbl = np.zeros(shape)
        # ipdb.set_trace()
        
        if os.path.exists(lbl_dir + 'liver' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'liver' + '.nii.gz')
            organ_lbl[array > 0] = 1
        if os.path.exists(lbl_dir + 'pancreas' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'pancreas' + '.nii.gz')
            organ_lbl[array > 0] = 2
        if os.path.exists(lbl_dir + 'kidney_left' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_left' + '.nii.gz')
            organ_lbl[array > 0] = 3
        if os.path.exists(lbl_dir + 'kidney_right' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_right' + '.nii.gz')
            organ_lbl[array > 0] = 3
        if os.path.exists(lbl_dir + 'liver_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'liver_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 4
        if os.path.exists(lbl_dir + 'pancreas_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'pancreas_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 5
        if os.path.exists(lbl_dir + 'pancreas_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 6

        if os.path.exists(lbl_dir + 'plaque' + '.nii.gz'):

            # ipdb.set_trace()
            array, mata_infomation = self._loader(lbl_dir + 'plaque' + '.nii.gz')
            organ_lbl[array==1] = 0
            organ_lbl[array==2] = 0
            organ_lbl[array==3] = 1
            organ_lbl[array==4] = 1

        return organ_lbl, mata_infomation

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        # print('file_name', d['name'])
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]

        return d

class HealthyOnlyDataset(Dataset):
    def __init__(self, base_dataset, max_retry: int = 20):
        self.base = base_dataset
        self.max_retry = max_retry

    def _is_healthy(self, lbl):
        if isinstance(lbl, torch.Tensor):
            return torch.count_nonzero(lbl) == 0
        return (lbl == 0).all()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        for _ in range(self.max_retry):
            sample = self.base[idx]

            # ---------- 情况 1：返回的是 patch 列表 ----------
            if isinstance(sample, list):
                if all(self._is_healthy(s["label"]) for s in sample):
                    return sample
            # ---------- 情况 2：单 patch ----------
            else:
                if self._is_healthy(sample["label"]):
                    return sample

            # 若不合格，随机换 idx 重试
            idx = random.randint(0, len(self.base) - 1)

        raise RuntimeError(
            f"[HealthyOnlyDataset] 连续 {self.max_retry} 次都抽不到纯健康 patch，"
            "请检查 roi_x/y/z 或增大 max_retry。"
        )

def _build_transforms(args, healthy: bool = False):
    """
    沿用原 preprocessing；仅在 healthy=True 时把 RandCropByPosNegLabeld
    改成 pos=0, neg=1 以优先抽负样本。
    """
    common = [
        LoadImaged_BodyMap(keys=["image"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=255.0,
            b_min=args.b_min, b_max=args.b_max, clip=True
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            mode=["minimum", "constant"],
        ),
    ]

    crop = RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=(args.roi_x, args.roi_y, args.roi_z),
        pos=0 if healthy else 4,
        neg=1 if healthy else 6,
        num_samples=args.num_samples, image_key="image",
        image_threshold=-1,
    )

    tail = [
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        ToTensord(keys=["image", "label"]),
    ]
    return Compose(common + [crop] + tail)
# ==========================================================================
# 🔸 NEW CODE END 🔸
# ==========================================================================

# --------------------------------------------------------------------------
# get_loader —— 仅在 train 分支做小改动
# --------------------------------------------------------------------------
def flatten_patch_collate(batch):
    """
    batch: List[  sample  ], 其中 sample 可能是
           - dict (单 patch 数据)
           - list[dict] (多 patch 数据，UniformDataset 输出)
    目标: 把所有 patch 扁平化到同一列表再丢给 list_data_collate，
         得到标准的 dict(tensor) 批次。
    """
    flat = []
    for s in batch:
        if isinstance(s, list):
            flat.extend(s)          # 展开 num_samples 个 patch
        else:
            flat.append(s)
    return list_data_collate(flat)


def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged_BodyMap(keys=["image"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=args.a_min,
            #     a_max=args.a_max,
            #     b_min=args.b_min,
            #     b_max=args.b_max,
            #     clip=True,
            # ),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode=["minimum", "constant"]),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            #     pos=20,
            #     neg=1,
            #     num_samples=args.num_samples,
            #     image_key="image",
            #     image_threshold=-1,
            # ),
            # RandCropByLabelClassesd(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
            #     ratios=[0, 0.5,0.5,1, 1],
            #     num_classes=5,
            #     num_samples=args.num_samples,
            #     image_key="image",
            #     image_threshold=-1,
            # ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=4,
                neg=6,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=2,
                neg=0,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    if args.phase == 'train':
        train_img=[]
        train_lbl=[]
        train_name=[]
        for line in open(os.path.join(args.data_txt_path,  args.dataset_list+'.txt')):
            name = line.strip().split('\t')[0]
            train_img.append(os.path.join(args.data_root_path, name + '/ct.nii.gz'))
            train_lbl.append(os.path.join(args.data_root_path, name + '/segmentations/'))
            # train_lbl.append(os.path.join(args.data_root_path, name + '/segmentation/'))
            train_name.append(name)
        data_dicts_train = [{'image': image, 'label': label, 'name': name}
                    for image, label, name in zip(train_img, train_lbl, train_name)]
        # print("data_dicts_train:",data_dicts_train)
        print('train len {}'.format(len(data_dicts_train)))



        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformCacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate, datasetkey=args.datasetkey)
            else:
                train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms, datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
        # import ipdb
        # ipdb.set_trace()
        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers,
                                    collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler, len(train_dataset)

    if args.phase == 'validation':
        val_img = []
        val_lbl = []
        val_name = []
        for item in args.dataset_list:
            for line in open(os.path.join(args.data_txt_path,  item, 'real_tumor_val_0.txt')):
                name = line.strip().split()[1].split('.')[0]
                val_img.append(os.path.join(args.data_root_path, line.strip().split()[0]))
                val_lbl.append(os.path.join(args.data_root_path, line.strip().split()[1]))
                val_name.append(name)
        data_dicts_val = [{'image': image, 'label': label, 'name': name}
                    for image, label, name in zip(val_img, val_lbl, val_name)]
        print('val len {}'.format(len(data_dicts_val)))

        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms, len(val_dataset)
    

def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key

if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()