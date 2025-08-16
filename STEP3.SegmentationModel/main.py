import os
import numpy as np
import json, time
from functools import partial
import nibabel as nb
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast #native AMP
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.transforms.transform import MapTransform
import sys
from os import environ

from monai.inferers import sliding_window_inference
# from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss, DiceCELoss,GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import load_decathlon_datalist
from monai.transforms import AsDiscrete,Activations,Compose

from monai import transforms, data
from monai_trainer import AMDistributedSampler, run_training
from optimizers.lr_scheduler import WarmupCosineSchedule,LinearWarmupCosineAnnealingLR
from networks.unetr import UNETR
from networks.swin3d_unetr import SwinUNETR
from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
import warnings
warnings.filterwarnings("ignore")
from monai.transforms import RandCropByLabelClassesd
import argparse
import random
# import importlib, distutils
# importlib.import_module("distutils.version")
parser = argparse.ArgumentParser(description='5fold cross val')

parser.add_argument('--syn',action='store_true')
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--logdir', default=None)
parser.add_argument('--save_checkpoint', action='store_true')
parser.add_argument('--max_epochs', default=3000, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--optim_lr', default=2e-4, type=float)
parser.add_argument('--synth_dir', default='/data/utsubo0/users/ruan/synth_alltumor_mcmaskinput_400step/', type=str)
# parser.add_argument('--synth_dir', default='/data/utsubo0/users/ruan/synth_alltumor_mcmask_transformer_400step/', type=str)
# parser.add_argument('--synth_dir', default='/data/utsubo0/users/ruan/synth_alltumor/', type=str)
parser.add_argument('--cache_dir', default='/data/utsubo0/users/ruan/tumor_cache/', type=str)
parser.add_argument('--optim_name', default='adamw', type=str)
parser.add_argument('--reg_weight', default=1e-5, type=float)

parser.add_argument('--noamp', action='store_true') #experimental
parser.add_argument('--val_every', default=200, type=int)
parser.add_argument('--val_overlap', default=0.5, type=float)
parser.add_argument('--cache_rate', default=0.5, type=float)

parser.add_argument('--distributed', default=False) #distributed multi gpu
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23457', type=str,  help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=4, type=int)


parser.add_argument('--model_name', default='unet', type=str)
parser.add_argument('--swin_type', default='base', type=str)
parser.add_argument('--tumor_type', default='tumor', type=str)
parser.add_argument('--organ_type', default='liver', type=str)
parser.add_argument('--organ_model', default='liver', type=str)
parser.add_argument('--diff_model', default=None, type=str)
parser.add_argument('--ddim_ts', default=50, type=int)
parser.add_argument('--fg_thresh', default=10, type=int)
parser.add_argument('--healthy_num', default=400, type=int)
parser.add_argument('--healthy_seed', default=0, type=int)
#segmentation flex params
parser.add_argument('--seg_block', default='basic_pre', type=str)
parser.add_argument('--seg_num_blocks', default = '1,2,2,4', type=str)
parser.add_argument('--seg_base_filters', default=16, type=int)
parser.add_argument('--seg_relu', default='relu', type=str)
parser.add_argument('--seg_lastnorm_init_zero', action='store_true')

parser.add_argument('--seg_mode', default=1, type=int)

parser.add_argument('--seg_use_se', action='store_true')
parser.add_argument('--seg_norm_name', default='instancenorm', type=str)
parser.add_argument('--seg_noskip', action='store_true')
parser.add_argument('--seg_aug_mode', default=0, type=int)
parser.add_argument('--seg_aug_noflip', action='store_true')

parser.add_argument('--seg_norm_mode', default=0, type=int)
parser.add_argument('--seg_crop_mode', default=0, type=int)

#unetr params
parser.add_argument('--pos_embedd', default='conv', type=str)
parser.add_argument('--norm_name', default='instance', type=str)
parser.add_argument('--num_steps', default=40000, type=int)
parser.add_argument('--eval_num', default=500, type=int)
parser.add_argument('--warmup_steps', default=500, type=int)
parser.add_argument('--num_heads', default=16, type=int)
parser.add_argument('--mlp_dim', default=3072, type=int)
parser.add_argument('--hidden_size', default=768, type=int)
parser.add_argument('--in_channels', default=1, type=int)
parser.add_argument('--out_channels', default=3, type=int)
parser.add_argument('--num_classes', default=3, type=int)
parser.add_argument('--res_block', action='store_true')
parser.add_argument('--conv_block', action='store_true')
parser.add_argument('--roi_x', default=96, type=int)
parser.add_argument('--roi_y', default=96, type=int)
parser.add_argument('--roi_z', default=96, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--decay', default=1e-5, type=float)
parser.add_argument('--lrdecay', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--amp_scale', action='store_true')
parser.add_argument('--opt_level', default='O2', type=str)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lrschedule', default='warmup_cosine', type=str)
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--resume_ckpt', action='store_true')
parser.add_argument('--pretrained_dir', default=None, type=str)
parser.add_argument('--data_root', default=None, type=str)
parser.add_argument('--healthy_data_root', default=None, type=str)
parser.add_argument('--datafold_dir', default=None, type=str)
parser.add_argument('--cache_num', default=200, type=int)

parser.add_argument('--use_pretrained', action='store_true')
# parser.add_argument('--synth_root', default=None, type=str)
parser.add_argument('--is_save_synth', action='store_true')
import ipdb
class AddTumorForegroundd(MapTransform):
    """
    非侵入式 remap:
    从原始 label 创建一个新的字段 'label_tumor'，
    其中 tumor 对应的 index 为 2 和 3 映射为1，其它（包括肾脏）均映射为0，
    保持原始 label 不变。
    """
    def __call__(self, data):
        d = dict(data)

        label = d["label"]
        # 如果 label 中 tumor index 为2或3，则输出1，其它为0
        label_tumor = np.where((label == 2) | (label == 3), 1, 0).astype(label.dtype)
        d[f"label_tumor"] = label_tumor
        # ipdb.set_trace()
        return d
class RandCropByPosNegLabeld_select(transforms.RandCropByPosNegLabeld):
    def __init__(self, keys, label_key, spatial_size, 
                 pos=1.0, neg=1.0, num_samples=1, 
                 image_key=None, image_threshold=0.0, allow_missing_keys=True,
                   fg_thresh=0):
        super().__init__(keys=keys, label_key=label_key, spatial_size=spatial_size, 
                 pos=pos, neg=neg, num_samples=num_samples, 
                 image_key=image_key, image_threshold=image_threshold, allow_missing_keys=allow_missing_keys)
        self.fg_thresh = fg_thresh

    def R2voxel(self,R):
        return (4/3*np.pi)*(R)**(3)

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']
        d.pop('name')
        if 'kidney_label' in data_name or 'liver_label' in data_name or 'pancreas_label' in data_name:
            flag=0
            while 1:
                flag+=1
                d_crop = super().__call__(d)
                pixel_num = (d_crop[0]['label']>0).sum()
                # print(pixel_num)
                if pixel_num > self.R2voxel(self.fg_thresh):
                    break
                if flag>5 and pixel_num > self.R2voxel(max(self.fg_thresh-5, 5)):
                    break
                if flag>10 and pixel_num > self.R2voxel(max(self.fg_thresh-10, 5)):
                    break
                if flag>15 and pixel_num > self.R2voxel(max(self.fg_thresh-15, 5)):
                    break
                if flag>20 and pixel_num > self.R2voxel(max(self.fg_thresh-20, 5)):
                    break
                if flag>25 and pixel_num > self.R2voxel(max(self.fg_thresh-25, 5)):
                    break
                if flag>30:
                    break
        else:
            d_crop = super().__call__(d)
        d_crop[0]['name'] = data_name

        return d_crop

class LoadImage_train(MapTransform):
    def __init__(self,organ_type):
        self.reader1 = transforms.LoadImaged(keys=["image", "label"])
        self.organ_type = organ_type

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']

        if (not 'kidney_label' in data_name) and self.organ_type == 'kidney':
            d = self.reader1.__call__(d)
            # d['label'][d['label']==3] = 1
        elif ('kidney_label' in data_name) and self.organ_type == 'kidney':
            d = self.reader1.__call__(d)
            d['label'][d['label']>0] = 1
            
        else :
            d = self.reader1.__call__(d)

        return d
    
class LoadImage_val(transforms.LoadImaged):
    def __init__(self, keys, *args,**kwargs, ):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']

        d = super().__call__(d)
        d['label'][d['label']==3] = 1

        return d
    
def _get_transform(args):

    train_transform = transforms.Compose(
    [
        LoadImage_train(args.organ_type),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
        # transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[128,128,128]),
        # transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[32,32,32]),
           # for kidney
        # AddTumorForegroundd(keys=["label"]),
        # RandCropByLabelClassesd(
        #     keys=["image", "label"],
        #     label_key="label",
        #     spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
        #     ratios=[0.5, 1, 2, 2, 2, 2, 2, 2] ,
        #     num_classes=8,
        #     num_samples=1,
        #     image_key="image",
        #     image_threshold=-1,
        # ), # 9
        # # for tumor

        RandCropByPosNegLabeld_select(
            keys=["image", "label", "name"],
            label_key="label",
            spatial_size=(96, 96, 96),
         #   spatial_size=(128,128,128),
            pos=2,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
            fg_thresh = args.fg_thresh,
        ),

        ## for kidney
        # RandCropByPosNegLabeld_select(
        #     keys=["image","label", "label_tumor"],
        #     label_key="label_tumor",
        #     spatial_size=(96, 96, 96),
        #     # spatial_size=(32,32,32),
        #     pos=2,
        #     neg=1,
        #     num_samples=1,
        #     image_key="image",
        #     image_threshold=0,
        #     fg_thresh = args.fg_thresh,
        # ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        transforms.ToTensord(keys=["image", "label"]),
    ]
    )

    val_transform = transforms.Compose(
        [
            LoadImage_val(keys=["image", "label", "organ_pseudo"]),
            transforms.AddChanneld(keys=["image", "label", "organ_pseudo"]),
            transforms.Orientationd(keys=["image", "label", "organ_pseudo"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label", "organ_pseudo"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            transforms.SpatialPadd(keys=["image", "label", "organ_pseudo"], mode=["minimum", "constant", "constant"], spatial_size=[96, 96, 96]),
            transforms.ToTensord(keys=["image", "label", "organ_pseudo"]),
        ]
    )
    
    return train_transform, val_transform

def get_all_file_paths(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.abspath(os.path.join(root, file)))

    return file_paths

def check_file_name_consistency(train_img_synth, train_lbl_synth, organ='not coronary'):
    # 遍历每个 organ 和 type 组合
    if organ == 'coronary':
        img_paths = train_img_synth
        lbl_paths = train_lbl_synth

        # 提取文件名
        img_files = set(os.path.basename(path) for path in img_paths)
        lbl_files = set(os.path.basename(path) for path in lbl_paths)

        # 检查文件名是否一致
        if img_files != lbl_files:
            print(f"Mismatch found at coronary:")
            print(f"Image files: {img_files}")
            print(f"Label files: {lbl_files}")
            return False  # 如果不一致，返回False
    else:
        for i in range(len(train_img_synth)):
            for j in range(len(train_img_synth[i])):
                img_paths = train_img_synth[i][j]
                lbl_paths = train_lbl_synth[i][j]

                # 提取文件名
                img_files = set(os.path.basename(path) for path in img_paths)
                lbl_files = set(os.path.basename(path) for path in lbl_paths)

                # 检查文件名是否一致
                if img_files != lbl_files:
                    print(f"Mismatch found at organ[{i}], type[{j}]:")
                    print(f"Image files: {img_files}")
                    print(f"Label files: {lbl_files}")
                    return False  # 如果不一致，返回False
    return True  # 全部一致则返回True

def shuffle_together(train_img_synth, train_lbl_synth):
    # 获取随机索引
    indices = list(range(len(train_img_synth)))
    random.shuffle(indices)
    # import ipdb
    # ipdb.set_trace()

    # 使用相同的随机索引重排两个列表
    train_img_synth = [train_img_synth[i] for i in indices]
    train_lbl_synth = [train_lbl_synth[i] for i in indices]

    return train_img_synth, train_lbl_synth

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')
    args.distributed = False

    # if args.distributed:
    #     args.ngpus_per_node = torch.cuda.device_count()
    #     print('Found total gpus', args.ngpus_per_node)
    #
    #     args.world_size = args.ngpus_per_node * args.world_size
    #     mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    #
    # else:
    #     # Simply call main_worker function
    #     main_worker(gpu=0, args=args)
    main_worker(gpu=0, args=args)

def main_worker(gpu, args):


    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True) #in new Pytorch/python labda functions fail to pickle with spawn
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)

    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu) #use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    print(args.rank, ' gpu', args.gpu)
    if args.rank==0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)

    roi_size = [args.roi_x, args.roi_y, args.roi_z]
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    
    data_root = args.data_root
    healthy_data_root = args.healthy_data_root
    datafold_dir = args.datafold_dir
    fold = args.fold
    tumor_type = args.tumor_type
    organ_type = args.organ_type
    if organ_type == 'liver':
        args.fg_thresh = 30
    elif organ_type == 'pancreas':
        args.fg_thresh = 15
    elif organ_type == 'kidney':
        args.fg_thresh = 25
    train_transform, val_transform = _get_transform(args)

    ## NETWORK
    if (args.model_name is None) or args.model_name == 'unet':
        from monai.networks.nets import UNet 
        model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    # out_channels=2, #for kidney tumor
                    out_channels=3, #for tumor
                    # out_channels=5, #for coronary


                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                )
        
    elif args.model_name == 'swinunetr':
        
        if args.swin_type == 'tiny':
            feature_size=12
        elif args.swin_type == 'small':
            feature_size=24
        elif args.swin_type == 'base':
            feature_size=48

        model = SwinUNETR_v2(in_channels=1,
                          out_channels=3,
                          img_size=(96, 96, 96),
                          feature_size=feature_size,
                          patch_size=2,
                          depths=[2, 2, 2, 2],
                          num_heads=[3, 6, 12, 24],
                          window_size=[7, 7, 7])
        
        if args.use_pretrained:
            pretrained_add = 'model_swinvit.pt'
            model.load_from(weights=torch.load(pretrained_add))
            print('Use pretrained ViT weights from: {}'.format(pretrained_add))
    elif args.model_name == 'nnunet':
        from monai.networks.nets import DynUNet
        from dynunet_pipeline.create_network import get_kernels_strides
        from dynunet_pipeline.task_params import deep_supr_num
        task_id = 'custom'
        kernels, strides = get_kernels_strides(task_id)
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=deep_supr_num[task_id],
        )
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))
        
    if args.resume_ckpt:
        model_dict = torch.load(args.pretrained_dir)
        model.load_state_dict(model_dict['state_dict'])
        print('Use pretrained weights')


    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    general_dice_loss = GeneralizedDiceLoss(to_onehot_y=True, softmax=True, smooth_nr=0, smooth_dr=1e-6)
    post_label = AsDiscrete(to_onehot=True, n_classes=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.num_classes)
    val_channel_names=['val_liver_dice', 'val_tumor_dice']

    print('Crop size', roi_size)

    train_img_real=[]
    train_lbl_real=[]
    train_name_real=[]
    train_img_healthy=[]
    train_lbl_healthy=[]
    train_name_healthy=[]


    train_txt = os.path.join(datafold_dir, 'real_{}_train_{}.txt'.format(tumor_type, fold))


    if os.path.exists(args.synth_dir) and not args.is_save_synth:
        if 'coronary' in datafold_dir:
            train_img_synth = []
            train_lbl_synth = []



            train_img_synth= get_all_file_paths( os.path.join(args.synth_dir,'coronary' , 'synth'))
            train_lbl_synth= get_all_file_paths( os.path.join(args.synth_dir,'coronary' , 'lbl'))
            # import ipdb
            # ipdb.set_trace()
            shuffle_together(train_img_synth, train_lbl_synth)
            assert check_file_name_consistency(train_img_synth, train_lbl_synth,organ='coronary')
            print('Synthetic {} data for coronary loaded'.format(len(train_img_synth)))
        else:

            train_img_synth = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
            train_lbl_synth = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
            organ_group = ['kidney', 'liver', 'pancreas']
            type_group = ['tiny', 'small', 'medium', 'large']
            for i in range(len(organ_group)):
                for j in range(len(type_group)):
                    train_img_synth[i][j]= get_all_file_paths( os.path.join(args.synth_dir,organ_group[i] ,type_group[j], 'synth'))
                    train_lbl_synth[i][j]= get_all_file_paths( os.path.join(args.synth_dir,organ_group[i] ,type_group[j], 'label'))
                    shuffle_together(train_img_synth[i][j], train_lbl_synth[i][j])
            assert check_file_name_consistency(train_img_synth, train_lbl_synth)
            print('Synthetic data for tumor loaded')
    print("is_save_synth", args.is_save_synth)




    if 'coronary' in datafold_dir:
        for line in open(os.path.join(datafold_dir, 'real_lesion_train_{}.txt'.format(args.fold))):
            name = line.strip().split('\t')[0]

            train_img_real.append(args.data_root + name+ '/ct.nii.gz')
            train_lbl_real.append(args.data_root + name+ '/segmentations/plaque.nii.gz')
            train_name_real.append(name)

            synth_threshold = 0

            # add synth data randomly
        train_img_healthy = train_img_synth
        train_lbl_healthy = train_lbl_synth
        train_name_healthy = train_lbl_synth

            # if random.random() > synth_threshold:
            #     if not os.path.exists(args.synth_dir) and args.is_save_synth:
            #         pass
            #     else:
            #
            #
            #         train_img_healthy.append(train_img_synth.pop())
            #         train_lbl_healthy.append(train_lbl_synth.pop())
            #         train_name_healthy.append(train_lbl_healthy[-1])
    else:
        for line in open(train_txt):
            name = line.strip().split()[1].split('.')[0]
            # if  not os.path.exists(args.synth_dir) and args.is_save_synth:
            if  args.is_save_synth:

                if 'kidney_label' in name or 'liver_label' in name or 'pancreas_label' in name:
                    train_img_healthy.append(healthy_data_root + line.strip().split()[0])
                    train_lbl_healthy.append(healthy_data_root + line.strip().split()[1])
                    train_name_healthy.append(name)
                else:
                    train_img_real.append(line.strip().split()[0])
                    train_lbl_real.append(line.strip().split()[1])
                    ##original reading
                    # train_img_healthy.append(data_root + line.strip().split()[0])
                    # train_lbl_healthy.append(data_root + line.strip().split()[1])
                    train_name_real.append(name)
            else:
                if 'kidney_label' in name or 'liver_label' in name or 'pancreas_label' in name:
                    pass
                else:
                    train_img_real.append(line.strip().split()[0])
                    train_lbl_real.append(line.strip().split()[1])
                    train_name_real.append(name)
            synth_threshold = 0.0
        #
        # add synth data randomly
            if not args.is_save_synth:
                if random.random() > synth_threshold:
                    if not os.path.exists(args.synth_dir) and args.is_save_synth:
                        pass
                    else:
                        tumor_types = ['tiny', 'small','medium', 'large']

                        tumor_probs = np.array([0.5, 0.2,0.2,0.1])
                        # tumor_probs = np.array([0.7, 0.1,0.1,0.1])
                        organ_group = ['kidney', 'liver', 'pancreas']
                        synthetic_tumor_type = np.random.choice(tumor_types, p=tumor_probs.ravel())
                        train_img_healthy.append(train_img_synth[organ_group.index(args.organ_type)][tumor_types.index(synthetic_tumor_type)].pop())
                        train_lbl_healthy.append(train_lbl_synth[organ_group.index(args.organ_type)][tumor_types.index(synthetic_tumor_type)].pop())
                        train_name_healthy.append(train_lbl_healthy[-1])



##### for tumor

    # import ipdb
    # ipdb.set_trace()

    print('train len real {}'.format(len(train_img_real)))
    print('train len healthy {}'.format(len(train_img_healthy)))
    train_img = train_img_real + train_img_healthy
    train_lbl = train_lbl_real + train_lbl_healthy
    train_name = train_name_real + train_name_healthy
    data_dicts_train = [{'image': image, 'label': label,"label_tumor":label, 'name': name}
            for image, label, name in zip(train_img, train_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))
    # import ipdb
    # ipdb.set_trace()

    val_img=[]
    val_lbl=[]
    val_name=[]
    val_pseudo_lbl=[]
    # for tumor
    for line in open(os.path.join(datafold_dir, 'real_{}_val_{}.txt'.format(tumor_type, fold))):
        if 'coronary' in datafold_dir:
            name = line.strip().split('\t')[0]
            val_img.append(args.data_root + name+ '/ct.nii.gz')
            val_lbl.append(args.data_root + name+ '/segmentations/plaque.nii.gz')
            val_pseudo_lbl.append('organ_pseudo_swin_new/{}/'.format(organ_type))
            val_name.append(name)

        else:

            name = line.strip().split()[1].split('.')[0]
            val_img.append(line.strip().split()[0])
            val_lbl.append(line.strip().split()[1])

            ##original reading
            # val_img.append(data_root + line.strip().split()[0])
            # val_lbl.append(data_root + line.strip().split()[1])
            val_pseudo_lbl.append('organ_pseudo_swin_new/{}/'.format(organ_type) + os.path.basename(line.strip().split()[1]))
            val_name.append(name)
    data_dicts_val = [{'image': image, 'label': label, 'organ_pseudo': organ_pseudo, 'name': name}
                for image, label, organ_pseudo, name in zip(val_img, val_lbl, val_pseudo_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))
    #
    # if 'coronary' in datafold_dir:
    #     train_img = train_img_real + val_img
    #     train_lbl = train_lbl_real + val_lbl
    #     train_name = train_name_real + val_name
    #     data_dicts_train = [{'image': image, 'label': label,"label_tumor":label, 'name': name}
    #                         for image, label, name in zip(train_img, train_lbl, train_name)]
    # print('train len {}'.format(len(data_dicts_train)))
    # ipdb.set_trace()

    
    val_shape_dict = {}
    for d in data_dicts_val:
        imagepath = d["image"]
        imagename = imagepath.split('/')[-1]
        imgnb = nb.load(imagepath)
        val_shape_dict[imagename] = [imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]]


    # 3) 指定一个磁盘缓存目录
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # 4) 创建一个缓存数据集
    train_ds = data.CacheDataset(data=data_dicts_train, transform=train_transform, cache_rate=args.cache_rate)

    train_sampler = AMDistributedSampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, sampler=train_sampler, pin_memory=True)


    val_ds = data.Dataset(data=data_dicts_val, transform=val_transform)
    val_sampler = AMDistributedSampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers//2, sampler=val_sampler, pin_memory=True)

    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor=model,  overlap=args.val_overlap, mode='gaussian')

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)



    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))


    model.cuda(args.gpu)
    

    
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name=='batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu) #??

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)


    # if args.optim_name == 'adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    # elif args.optim_name == 'adamw':
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    # elif args.optim_name=='sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=0.99, nesterov=True, weight_decay=args.reg_weight) #momentum 0.99, nestorov=True, following nnUnet
    # else:
    #     raise ValueError('Unsupported optim_name' + str(args.optim_name))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )

    # if args.lrschedule == 'warmup_cosine':
    #     scheduler = LinearWarmupCosineAnnealingLR(
    #         optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
    #     )
    #
    #
    # elif args.lrschedule == 'cosine_anneal':
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    #     if args.checkpoint is not None:
    #         scheduler.step(epoch=start_epoch)
    #
    # else:
    #     scheduler = None



    accuracy = run_training(model=model,
                             train_loader=train_loader,
                             val_loader=val_loader,
                             optimizer=optimizer,
                             loss_func=dice_loss,
                             # loss_func=general_dice_loss,
                             args=args,
                             model_inferer=model_inferer,
                             scheduler=lr_scheduler,
                             start_epoch=start_epoch,
                             val_channel_names=val_channel_names,
                             val_shape_dict=val_shape_dict,
                             post_label=post_label,
                             post_pred=post_pred, val_acc_max = best_acc)

    return accuracy


if __name__ == '__main__':
    main()
