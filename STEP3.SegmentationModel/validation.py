import os, time, csv
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from scipy.ndimage import label
from functools import partial
from surface_distance import compute_surface_distances,compute_surface_dice_at_tolerance
import monai
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist
from monai.transforms import AsDiscrete,AsDiscreted,Compose,Invertd,SaveImaged
from monai import transforms, data
from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
import nibabel as nib

import warnings
warnings.filterwarnings("ignore")

import argparse
import ipdb
parser = argparse.ArgumentParser(description='liver tumor validation')

# file dir
parser.add_argument('--data_root', default=None, type=str)
parser.add_argument('--datafold_dir', default=None, type=str)
parser.add_argument('--tumor_type', default='early', type=str)
parser.add_argument('--organ_type', default='liver', type=str)
parser.add_argument('--fold', default=0, type=int)

parser.add_argument('--save_dir', default='out', type=str)
parser.add_argument('--checkpoint', action='store_true')

parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--feature_size', default=16, type=int)
parser.add_argument('--val_overlap', default=0.75, type=float)
parser.add_argument('--num_classes', default=3, type=int)

parser.add_argument('--model', default='unet', type=str)
parser.add_argument('--swin_type', default='base', type=str)

def organ_region_filter_out(organ_mask, tumor_mask):
    ## dialtion
    organ_mask = ndimage.binary_closing(organ_mask, structure=np.ones((5,5,5)))
    organ_mask = ndimage.binary_dilation(organ_mask, structure=np.ones((5,5,5)))
    ## filter out
    tumor_mask = organ_mask * tumor_mask

    return tumor_mask

def denoise_pred(pred: np.ndarray):
    """
    # 0: background, 1: liver, 2: tumor.
    pred.shape: (3, H, W, D)
    """
    denoise_pred = np.zeros_like(pred)


    denoise_pred[1, ...] = pred[1, ...]

    denoise_pred[2, ...] = pred[1, ...] * pred[2,...]

    denoise_pred[0,...] = 1 - np.logical_or(denoise_pred[1,...], denoise_pred[2,...])

    return denoise_pred

def cal_dice(pred, true):
    import ipdb
    # ipdb.set_trace()
    intersection = np.sum(pred[true==1]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    # print("sum_true: ", np.sum(true==1))
    # print("intersection: ", intersection)
    # print("sum_pred+sum_true: ", (np.sum(pred) + np.sum(true)))
    # print("dice: ", dice)
    # ipdb.set_trace()
    return dice

def cal_dice_nsd(pred, truth, spacing_mm=(1,1,1), tolerance=2):
    dice = cal_dice(pred, truth)
    # cal nsd
    surface_distances = compute_surface_distances(truth.astype(bool), pred.astype(bool), spacing_mm=spacing_mm)
    nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)

    return (dice, nsd)


def _get_model(args):
    # inf_size = [32, 32, 32]
    inf_size = [96, 96, 96]
    # inf_size = [128,128,128]
    print(args.model)
    if args.model == 'swinunetr':
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
        
    elif args.model == 'unet':
        from monai.networks.nets import UNet
        # ## for tumor
        model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    # out_channels=2, #for kidney
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                )
        # ## for coronary plaque
        # model = UNet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=5,
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),
        #     num_res_units=2,
        # )
    elif args.model == 'nnunet':
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
        raise ValueError('Unsupported model ' + str(args.model))


    if args.checkpoint:
        checkpoint = torch.load(os.path.join(args.log_dir, 'model_final.pt'))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)
        print('Use logdir weights')
    else:
        model_dict = torch.load(os.path.join(args.log_dir, 'model_final.pt'))
        model.load_state_dict(model_dict['state_dict'])
        print('Use logdir weights')

    model = model.cuda()
    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor=model,  overlap=args.val_overlap, mode='gaussian')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    return model, model_inferer

def _get_loader(args):
    val_org_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label", "organ_pseudo"]),
            transforms.AddChanneld(keys=["image", "label", "organ_pseudo"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            # transforms.SpatialPadd(keys=["image"], mode="minimum", spatial_size=[96, 96, 96]),
            transforms.SpatialPadd(keys=["image"], mode="minimum", spatial_size=[128,128,128]),
            # transforms.SpatialPadd(keys=["image"], mode="minimum", spatial_size=[32, 32, 32]),
            transforms.ToTensord(keys=["image", "label", "organ_pseudo"]),
        ]
    )
    val_img=[]
    val_lbl=[]
    val_name=[]
    val_pseudo_lbl = []

    ### test training data
    # for line in open(os.path.join(args.datafold_dir, 'real_{}_train_{}.txt'.format(args.tumor_type, args.fold))):
    #     name = line.strip().split()[1].split('.')[0]
    #     if 'kidney_label' in name or 'liver_label' in name or 'pancreas_label' in name:
    #         pass
    #     else:
    #         # name = line.strip().split()[1].split('.')[0]
    #     # val_img.append(args.data_root + line.strip().split()[0])
    #     # val_lbl.append(args.data_root + line.strip().split()[1])
    #         val_img.append(line.strip().split()[0])
    #         val_lbl.append(line.strip().split()[1])
    #         val_pseudo_lbl.append('organ_pseudo_swin_new/'+args.organ_type + '/' + os.path.basename(line.strip().split()[1]))
    #         val_name.append(name)

    import ipdb


    ## original val data
    # for line in open(os.path.join(args.datafold_dir, 'real_{}_train_{}.txt'.format(args.tumor_type, args.fold))):
    for line in open(os.path.join(args.datafold_dir, 'real_{}_val_{}.txt'.format(args.tumor_type, args.fold))):
        # ipdb.set_trace()
        # name = line.strip().split()[1].split('.')[0] ### for tumor
        if "/data/utsubo0" in line:

            name = line.strip().split('\t')[0] ### for coronary plaque
            #
            # val_img.append(args.data_root + line.strip().split()[0])
            # val_lbl.append(args.data_root + line.strip().split()[1])

        # ### for coronary plaque
        # val_img.append(args.data_root + name+ '/ct.nii.gz')
        # val_lbl.append(args.data_root + name+ '/segmentations/plaque.nii.gz')
        # val_name.append(name)

            ## for tumor
            val_img.append(line.strip().split()[0])
            val_lbl.append(line.strip().split()[1])
            val_pseudo_lbl.append('organ_pseudo_swin_new/'+args.organ_type + '/' + os.path.basename(line.strip().split()[1]))
            val_name.append(name)
    # change organ_preudo to val_lbl
    data_dicts_val = [{'image': image, 'label': label, 'organ_pseudo': organ_pseudo, 'name': name}
                      for image, label, organ_pseudo, name in zip(val_img, val_lbl, val_lbl, val_name)]
    # data_dicts_val = [{'image': image, 'label': label, 'organ_pseudo': organ_pseudo, 'name': name}
    #             for image, label, organ_pseudo, name in zip(val_img, val_lbl, val_pseudo_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))

    val_org_ds = data.Dataset(data_dicts_val, transform=val_org_transform)
    val_org_loader = data.DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4, sampler=None, pin_memory=True)

    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=val_org_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        ## for tumor
        AsDiscreted(keys="pred", argmax=True, to_onehot=3),
        AsDiscreted(keys="label", to_onehot=3),
        AsDiscreted(keys="organ_pseudo", to_onehot=3),
        # AsDiscreted(keys="pred", argmax=True, to_onehot=5), # for coronary
        # AsDiscreted(keys="label", to_onehot=5),# for coronary
        # AsDiscreted(keys="organ_pseudo", to_onehot=5),# for coronary
    ])
    
    return val_org_loader, post_transforms

def main():
    args = parser.parse_args()
    model_name = args.log_dir.split('/')[-1]
    args.model_name = model_name
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    torch.cuda.set_device(0) #use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    ## loader and post_transform
    val_loader, post_transforms = _get_loader(args)

    ## NETWORK
    model, model_inferer = _get_model(args)

    organ_dice = []
    organ_nsd  = []
    tumor_dice = []
    tumor_nsd  = []
    cal_dice = []
    cal_nsd = []
    noncal_dice = []
    noncal_nsd = []
    header = ['name', 'organ_dice', 'organ_nsd', 'tumor_dice', 'tumor_nsd']
    rows = []

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            val_inputs = val_data["image"].cuda()
            # for tumor
            name = val_data['label_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
            ## for coronary
            # name = val_data['name'][0]
            # import ipdb
            # ipdb.set_trace()
            original_affine = val_data["label_meta_dict"]["affine"][0].numpy()
            pixdim = val_data['label_meta_dict']['pixdim'].cpu().numpy()
            spacing_mm = tuple(pixdim[0][1:4])
            # breakpoint()
            # ipdb.set_trace()

            # ## for tumor
            val_data['label'][val_data['label']==2] = 2
            val_data['label'][val_data['label']==3] = 2
            val_data['label'][val_data['label']==4] = 2
            val_data['label'][val_data['label']==5] = 2
            val_data['label'][val_data['label']==6] = 2
            val_data['label'][val_data['label']==7] = 2
            val_data['organ_pseudo'][val_data['organ_pseudo']==2] = 2
            val_data['organ_pseudo'][val_data['organ_pseudo']==3] = 2
            val_data['organ_pseudo'][val_data['organ_pseudo']==4] = 2
            val_data['organ_pseudo'][val_data['organ_pseudo']==5] = 2
            val_data['organ_pseudo'][val_data['organ_pseudo']==6] = 2
            val_data['organ_pseudo'][val_data['organ_pseudo']==7] = 2

            # #for kidney
            # if args.organ_type == 'kidney':
            #     val_data['label'][val_data['label']==1] = 0
            #     val_data['label'][val_data['label']==2] = 1
            #     val_data['organ_pseudo'][val_data['organ_pseudo']==1] = 0
            #     val_data['organ_pseudo'][val_data['organ_pseudo']==2] = 1


            # val_data['label'][val_data['label']==3] = 1
            val_data["pred"] = model_inferer(val_inputs)
            # ipdb.set_trace()
            # ipdb.set_trace()
            val_data = [post_transforms(i) for i in data.decollate_batch(val_data)]
            val_outputs, val_labels, val_organ_pseudo = val_data[0]['pred'], val_data[0]['label'], val_data[0]['organ_pseudo']

            
            # val_outpus.shape == val_labels.shape  (3, H, W, Z)

            # ##for coronary
            # print("val_label_cal_num:",(val_labels[3]==1).sum())
            # print("val_label_noncal_num:",(val_labels[4]==1).sum())


            # indices_cal = torch.nonzero(val_labels[3]==1, as_tuple=False)
            # indices_noncal = torch.nonzero(val_labels[4]==1, as_tuple=False)
            #
            #     # return None
            # # Calculate center of mass as the mean of the coordinates
            # center_point_cal = indices_cal.float().mean(dim=0)
            # center_point_noncal = indices_noncal.float().mean(dim=0)
            # print(f"cal label Center Point: {center_point_cal.tolist()}")
            # print(f"noncal label Center Point: {center_point_noncal.tolist()}")

            # ## for coronary
            # print("val_output_cal_num:",(val_outputs[3]>=0.5).sum())
            # print("val_output_noncal_num:",(val_outputs[4]>=0.5).sum())


            # indices_cal = torch.nonzero(val_labels[3]==1, as_tuple=False)
            # indices_noncal = torch.nonzero(val_labels[4]==1, as_tuple=False)
            #
            #
            # # return None
            # # Calculate center of mass as the mean of the coordinates
            # center_point_cal = indices_cal.float().mean(dim=0)
            # center_point_noncal = indices_noncal.float().mean(dim=0)
            # print(f"cal label Center Point: {center_point_cal.tolist()}")
            # print(f"noncal label Center Point: {center_point_noncal.tolist()}")



            val_outputs, val_labels = val_outputs.detach().cpu().numpy(), val_labels.detach().cpu().numpy()
            val_organ_pseudo = val_organ_pseudo.detach().cpu().numpy()

            # ipdb.set_trace()
            # val_outputs[1, ...] = val_organ_pseudo[1, ...]

            # val_outputs = denoise_pred(val_outputs)
            # ipdb.set_trace()
            # ## for coronary
            # print("--------cal_dice_nsd--------")
            # current_cal_dice, current_cal_nsd = cal_dice_nsd(val_outputs[3,...], val_labels[3,...], spacing_mm=spacing_mm)
            # print("--------noncal_dice_nsd--------")
            # current_noncal_dice, current_noncal_nsd = cal_dice_nsd(val_outputs[4,...], val_labels[4,...], spacing_mm=spacing_mm)

            #
            ## for tumor
            current_liver_dice, current_liver_nsd = cal_dice_nsd(val_outputs[1,...], val_labels[1,...], spacing_mm=spacing_mm)
            current_tumor_dice, current_tumor_nsd = cal_dice_nsd(val_outputs[2,...], val_labels[2,...], spacing_mm=spacing_mm)

            # ## for kidney
            #
            #
            #
            # current_tumor_dice, current_tumor_nsd = cal_dice_nsd(val_outputs[1,...], val_labels[1,...], spacing_mm=spacing_mm)

            ## for tumor
            organ_dice.append(current_liver_dice)
            organ_nsd.append(current_liver_nsd)

            #
            # ## for coronary
            # if val_labels[3,...].sum() >0 :
            #     cal_dice.append(current_cal_dice)
            #     cal_nsd.append(current_cal_nsd)
            # if val_labels[4,...].sum() >0:
            #     noncal_dice.append(current_noncal_dice)
            #     noncal_nsd.append(current_noncal_nsd)

            ## for tumor
            tumor_dice.append(current_tumor_dice)
            tumor_nsd.append(current_tumor_nsd)
            #
            #
            #
            # # row = [name, current_tumor_dice, current_tumor_nsd]
            # fortumor
            row = [name, current_liver_dice, current_liver_nsd, current_tumor_dice, current_tumor_nsd]
            # # for kidney
            # rows = [name, current_tumor_dice, current_tumor_nsd]

            # ## for coronary
            # row = [name, current_cal_dice, current_cal_nsd, current_noncal_dice, current_noncal_nsd]
            rows.append(row)
            # for tumor
            print(name, val_outputs[0].shape, \
                  # 'dice: [{:.3f} ]; nsd: [ {:.3f}]'.format(current_tumor_dice,  current_tumor_nsd), \
                  'dice: [{:.3f}  {:.3f}]; nsd: [{:.3f}  {:.3f}]'.format(current_liver_dice, current_tumor_dice, current_liver_nsd, current_tumor_nsd), \
                'time {:.2f}s'.format(time.time() - start_time))

            # ## for kidney
            # print(name, val_outputs[0].shape, \
            #       # 'dice: [{:.3f} ]; nsd: [ {:.3f}]'.format(current_tumor_dice,  current_tumor_nsd), \
            #       'dice: [{:.3f} ]; nsd: [{:.3f} ]'.format( current_tumor_dice,  current_tumor_nsd), \
            #       'time {:.2f}s'.format(time.time() - start_time))

            #
            ##for coronary

            # print(name, val_outputs[0].shape, \
            #       # 'dice: [{:.3f} ]; nsd: [ {:.3f}]'.format(current_tumor_dice,  current_tumor_nsd), \
            #       'dice: [{:.3f}  {:.3f}]; nsd: [{:.3f}  {:.3f}]'.format(current_cal_dice,current_noncal_dice,current_cal_nsd,current_noncal_nsd), \
            #       'time {:.2f}s'.format(time.time() - start_time))

            # save the prediction
            output_dir = os.path.join(args.save_dir, args.model_name, str(args.val_overlap), 'pred')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # val_outputs = np.argmax(val_outputs, axis=0)
            val_outputs_ = np.zeros_like(val_outputs[0])
            val_outputs_[val_outputs[1]==1] = 1
            val_outputs_[val_outputs[2]==1] = 2
            # ## for coronary
            # val_outputs_[val_outputs[3]==1] = 3
            # val_outputs_[val_outputs[4]==1] = 4

            nib.save(
                nib.Nifti1Image(val_outputs_.astype(np.uint8), original_affine), os.path.join(output_dir, f'{name}.nii.gz')
            )

        ## tumor
        print("liver dice:", np.mean(organ_dice))
        print("liver nsd:", np.mean(organ_nsd))
        print("tumor dice:", np.mean(tumor_dice))
        print("tumor nsd",np.mean(tumor_nsd))
        rows.append(['average', np.mean(organ_dice), np.mean(organ_nsd), np.mean(tumor_dice), np.mean(tumor_nsd)])

        #
        # # coroaary
        # print("cal dice:", np.mean(cal_dice))
        # print("cal nsd:", np.mean(cal_nsd))
        # print("noncal dice:", np.mean(noncal_dice))
        # print("noncal nsd",np.mean(noncal_nsd))
        # rows.append(['average', np.mean(cal_dice), np.mean(cal_nsd), np.mean(noncal_dice), np.mean(noncal_nsd)])

        # save metrics to cvs file
        csv_save = os.path.join(args.save_dir, args.model_name, str(args.val_overlap))
        if not os.path.exists(csv_save):
            os.makedirs(csv_save)
        csv_name = os.path.join(csv_save, 'metrics.csv')
        with open(csv_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

if __name__ == "__main__":
    main()
