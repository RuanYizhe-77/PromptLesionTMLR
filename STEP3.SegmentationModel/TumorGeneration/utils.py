### Tumor Generateion
import random
import cv2
import elasticdeform
import numpy as np
from scipy.ndimage import gaussian_filter
from TumorGeneration.ldm.ddpm.ddim import DDIMSampler
from .ldm.vq_gan_3d.model.vqgan import VQGAN
import matplotlib.pyplot as plt
import SimpleITK as sitk
from .ldm.ddpm import Unet3D, GaussianDiffusion, Tester
from hydra import initialize, compose
import torch
import torch.nn.functional as F
import yaml

import ipdb

# Random select location for tumors.
def random_select(mask_scan, organ_type):
    # ipdb.set_trace()
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0].min(), np.where(np.any(mask_scan, axis=(0, 1)))[0].max()

    flag=0
    while 1:
        # ipdb.set_trace()
        if flag<=10:
            z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start
        elif flag>10 and flag<=20:
            z = round(random.uniform(0.2, 0.8) * (z_end - z_start)) + z_start
        elif flag>20 and flag<=30:
            z = round(random.uniform(0.1, 0.9) * (z_end - z_start)) + z_start
        else:
            z = round(random.uniform(0.0, 1.0) * (z_end - z_start)) + z_start
        liver_mask = mask_scan[..., z]

        if organ_type == 'liver':
            kernel = np.ones((5,5), dtype=np.uint8)
            liver_mask = cv2.erode(liver_mask, kernel, iterations=1)
        if (liver_mask == 1).sum() > 0:
            break
        flag+=1

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

def center_select(mask_scan):
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0].min(), np.where(np.any(mask_scan, axis=(0, 1)))[0].max()
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0].min(), np.where(np.any(mask_scan, axis=(1, 2)))[0].max()
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0].min(), np.where(np.any(mask_scan, axis=(0, 2)))[0].max()

    z = round(0.5 * (z_end - z_start)) + z_start
    x = round(0.5 * (x_end - x_start)) + x_start
    y = round(0.5 * (y_end - y_start)) + y_start

    xyz = [x, y, z]
    potential_points = xyz

    return potential_points

# generate the ellipsoid
def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out

def get_fixed_geo(mask_scan, tumor_type, organ_type):
    if tumor_type == 'large':
        enlarge_x, enlarge_y, enlarge_z = 280, 280, 280
    else:
        enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32
    # ipdb.set_trace()

    if tumor_type == 'tiny':
        num_tumor = random.randint(1,3)
        # ipdb.set_trace()
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            # ipdb.set_trace()
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # ipdb.set_trace()
            point = random_select(mask_scan, organ_type)
            # ipdb.set_trace()
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2
            # ipdb.set_trace()
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # ipdb.set_trace()


    if tumor_type == 'small':
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan, organ_type)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo


    if tumor_type == 'medium':
        num_tumor = 1
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan, organ_type)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'large':
        num_tumor = 1
        for _ in range(num_tumor):
            # Large tumor
            
            x = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            y = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            z = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            sigma = random.randint(5, 10)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            if organ_type == 'liver' or organ_type == 'kidney' :
                point = random_select(mask_scan, organ_type)
                new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
                x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
                y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
                z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            else:
                x_start, x_end = np.where(np.any(geo, axis=(1, 2)))[0].min(), np.where(np.any(geo, axis=(1, 2)))[0].max()
                y_start, y_end = np.where(np.any(geo, axis=(0, 2)))[0].min(), np.where(np.any(geo, axis=(0, 2)))[0].max()
                z_start, z_end = np.where(np.any(geo, axis=(0, 1)))[0].min(), np.where(np.any(geo, axis=(0, 1)))[0].max()
                geo = geo[x_start:x_end, y_start:y_end, z_start:z_end]

                point = center_select(mask_scan)

                new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
                x_low = new_point[0] - geo.shape[0]//2
                y_low = new_point[1] - geo.shape[1]//2
                z_low = new_point[2] - geo.shape[2]//2
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_low+geo.shape[0], y_low:y_low+geo.shape[1], z_low:z_low+geo.shape[2]] += geo
    # ipdb.set_trace()
    geo_mask = geo_mask[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]

    if ((tumor_type == 'medium') or (tumor_type == 'large')) and (organ_type == 'kidney'):

        geo_mask = (geo_mask * mask_scan) >=1
    else:
        geo_mask = (geo_mask * mask_scan) >=1

    return geo_mask
def synt_model_prepare(device, vqgan_ckpt='TumorGeneration/model_weight/AutoencoderModel.ckpt', diffusion_ckpt='TumorGeneration/model_weight/', label_idx=2, organ='liver'):
# def synt_model_prepare(device, vqgan_ckpt='TumorGeneration/model_weight/AutoencoderModelTumor.ckpt', diffusion_ckpt='TumorGeneration/model_weight/', label_idx=2, organ='liver'):
    with initialize(config_path="diffusion_config/"):
        cfg=compose(config_name="ddpm.yaml")
    print('diffusion_ckpt',diffusion_ckpt)
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt)
    vqgan = vqgan.to(device)
    vqgan.eval()
    
    # early_Unet3D = Unet3D(
    #         dim=cfg.diffusion_img_size,
    #         dim_mults=cfg.dim_mults,
    #         channels=cfg.diffusion_num_channels,
    #         out_dim=cfg.out_dim
    #         ).to(device)

## for promptlesion
    early_Unet3D = Unet3D(
        dim=cfg.diffusion_img_size,
        dim_mults=cfg.dim_mults,
        channels=17,
        out_dim=cfg.out_dim
    ).to(device)

    # early_diffusion = GaussianDiffusion(
    #         early_Unet3D,
    #         vqgan_ckpt= vqgan_ckpt, # cfg.vqgan_ckpt,
    #         image_size=cfg.diffusion_img_size,
    #         num_frames=cfg.diffusion_depth_size,
    #         channels=cfg.diffusion_num_channels,
    #         timesteps=400,
    #         # timesteps=4,
    #         loss_type=cfg.loss_type,
    #         device=device
    #         ).to(device)

    ## for promptlesion
    early_diffusion = GaussianDiffusion(
        early_Unet3D,
        vqgan_ckpt= vqgan_ckpt, # cfg.vqgan_ckpt,
        image_size=cfg.diffusion_img_size,
        num_frames=cfg.diffusion_depth_size,
        channels=17,
        timesteps=400,
        # timesteps=4,
        loss_type=cfg.loss_type,
        device=device
    ).to(device)
    ##tumor stage
    # noearly_Unet3D = Unet3D(
    #         dim=cfg.diffusion_img_size,
    #         dim_mults=cfg.dim_mults,
    #         channels=cfg.diffusion_num_channels,
    #         out_dim=cfg.out_dim
    #         ).to(device)
    #
    # noearly_diffusion = GaussianDiffusion(
    #         noearly_Unet3D,
    #         vqgan_ckpt= vqgan_ckpt,
    #         image_size=cfg.diffusion_img_size,.
    #         num_frames=cfg.diffusion_depth_size,
    #         channels=cfg.diffusion_num_channels,
    #         timesteps=200,
    #         loss_type=cfg.loss_type,
    #         device=device
    #         ).to(device)
    # print("diffusion_ckpt:"+diffusion_ckpt+'{}_400t_alltumor_ratiofix.pt'.format(organ)) ## fake 400
    print("diffusion_ckpt:"+diffusion_ckpt+'{}_400step_alltumor_mcmask-transformer.pt'.format(organ)) ## real 400
    # print("diffusion_ckpt:"+diffusion_ckpt+'{}_400step_alltumor_mcmaskinput.pt'.format(organ)) ## real 400
    early_tester = Tester(early_diffusion)
    # early_tester.load(diffusion_ckpt+'{}_early.pt'.format(organ), map_location=device)
    # early_tester.load(diffusion_ckpt+'{}_alltumor_mcmask_400t.pt'.format(organ), map_location=device)
    # early_tester.load(diffusion_ckpt+'{}_400t_alltumor_ratiofix.pt'.format(organ), map_location=device) ## fake 400
    early_tester.load(diffusion_ckpt+'{}_400step_alltumor_mcmask-transformer.pt'.format(organ), map_location=device) ## real 400
    # early_tester.load(diffusion_ckpt+'{}_400step_alltumor_mcmaskinput.pt'.format(organ), map_location=device) ## real 400

    # early_tester.load(diffusion_ckpt+'{}_early_400t_{}.pt'.format(organ,label_idx), map_location=device)
    # # tumor stage
    # noearly_checkpoint = torch.load(diffusion_ckpt+'{}_noearly.pt'.format(organ), map_location=device)
    # noearly_diffusion.load_state_dict(noearly_checkpoint['ema'])
    # noearly_sampler = DDIMSampler(noearly_diffusion, schedule="cosine")

    # return vqgan, early_tester, noearly_sampler
    return vqgan, early_tester
def calculate_multiclass_l1_loss(pred, target, mask):
    loss = torch.zeros(mask.shape[1])
    new_shape = [mask.shape[1]] + list(pred.shape)

    # 使用 torch.zeros() 而不是 torch.zeros_like() 创建具有新形状的张量
    pred_ = torch.zeros(new_shape, dtype=pred.dtype, device=pred.device)
    target_ = torch.zeros(new_shape, dtype=target.dtype, device=target.device)
    # pred_ = torch.zeros_like( [mask.shape[1]] + list(pred.shape))
    # target_ = torch.zeros_like( [mask.shape[1]] + list(target.shape))
    # ipdb.set_trace()
    count_ = torch.zeros(mask.shape[1])
    for i in range(mask.shape[1]):
        mask_i = mask[:,i,:,:,:].unsqueeze(1).float()
        count = mask_i.sum()
        count_[i]=count

        if count == 0:
            continue
        # ipdb.set_trace()
        pred_[i] = pred * mask_i
        target_[i] = target * mask_i
        # ipdb.set_trace()
        l1_loss = torch.abs(pred_[i] - target_[i])
        loss[i] = l1_loss.sum() / count
        # ipdb.set_trace()
    # ipdb.set_trace()
    return loss, pred_, target_,count_

#for alltumor-prompt

def synthesize_alltumor_with_mask(ct_volume, mask, vqgan, tester,idx,args):
    device=ct_volume.device
    mask_t = mask.clone()
    mask_shape = list(mask_t.shape)  # 将 torch.Size 转换为列表
    mask_shape[1] = 7
    mask_multi = torch.zeros(torch.Size(mask_shape)).cuda()
    mask_shape[1] = 6
    mask_multi_lesion = torch.zeros(torch.Size(mask_shape)).cuda()

    #
    # mask[:,1,:,:,:][mask_.squeeze(1)==2]= 1.0
    # mask[:,2,:,:,:][mask_.squeeze(1)==3]= 1.0
    # mask[:,3,:,:,:][mask_.squeeze(1)==4]= 1.0
    # mask[:,4,:,:,:][mask_.squeeze(1)==5]= 1.0
    # mask[:,5,:,:,:][mask_.squeeze(1)==6]= 1.0
    # mask[:,6,:,:,:][mask_.squeeze(1)==7]= 1.0

    mask_multi[:,1,:,:,:][mask_t.squeeze(1)==2]= 1.0
    mask_multi[:,2,:,:,:][mask_t.squeeze(1)==3]= 1.0
    mask_multi[:,3,:,:,:][mask_t.squeeze(1)==4]= 1.0
    mask_multi[:,4,:,:,:][mask_t.squeeze(1)==5]= 1.0
    mask_multi[:,5,:,:,:][mask_t.squeeze(1)==6]= 1.0
    mask_multi[:,6,:,:,:][mask_t.squeeze(1)==7]= 1.0
    mask_multi[:,0,:,:,:][mask_t.squeeze(1)==0]= 1.0
    mask_multi[:,0,:,:,:][mask_t.squeeze(1)==1]= 1.0
    mask_multi_lesion[:,0,:,:,:][mask_t.squeeze(1)==2]= 1.0
    mask_multi_lesion[:,1,:,:,:][mask_t.squeeze(1)==3]= 1.0
    mask_multi_lesion[:,2,:,:,:][mask_t.squeeze(1)==4]= 1.0
    mask_multi_lesion[:,3,:,:,:][mask_t.squeeze(1)==5]= 1.0
    mask_multi_lesion[:,4,:,:,:][mask_t.squeeze(1)==6]= 1.0
    mask_multi_lesion[:,5,:,:,:][mask_t.squeeze(1)==7]= 1.0

    ### add prompt label
    mask_shape[1] = 7
    mask_p = torch.zeros(torch.Size(mask_shape)).cuda()
    mask_p[:,1,:,:,:][mask_t.squeeze(1)==2]= 1.0
    mask_p[:,2,:,:,:][mask_t.squeeze(1)==3]= 1.0
    mask_p[:,3,:,:,:][mask_t.squeeze(1)==4]= 1.0
    mask_p[:,4,:,:,:][mask_t.squeeze(1)==5]= 1.0
    mask_p[:,5,:,:,:][mask_t.squeeze(1)==6]= 1.0
    mask_p[:,6,:,:,:][mask_t.squeeze(1)==6]= 1.0

    sum_channel_0 = mask_p[:, 1, :, :, :].sum(dim=(1, 2, 3))
    sum_channel_1 = mask_p[:, 2, :, :, :].sum(dim=(1, 2, 3))
    sum_channel_2 = mask_p[:, 3, :, :, :].sum(dim=(1, 2, 3))
    sum_channel_3 = mask_p[:, 4, :, :, :].sum(dim=(1, 2, 3))
    sum_channel_4 = mask_p[:, 5, :, :, :].sum(dim=(1, 2, 3))
    sum_channel_5 = mask_p[:, 6, :, :, :].sum(dim=(1, 2, 3))

    # 初始化标签为 0
    labels = torch.zeros(mask_shape[0], dtype=torch.long).cuda()

    labels = torch.where(sum_channel_0 > 0, torch.tensor(1).cuda(), labels)
    labels = torch.where(sum_channel_1 > 0, torch.tensor(2).cuda(), labels)
    labels = torch.where(sum_channel_2 > 0, torch.tensor(3).cuda(), labels)
    labels = torch.where(sum_channel_3 > 0, torch.tensor(4).cuda(), labels)
    labels = torch.where(sum_channel_4 > 0, torch.tensor(5).cuda(), labels)
    labels = torch.where(sum_channel_5 > 0, torch.tensor(6).cuda(), labels)
    if torch.rand(1).item() < 0.5:
        labels = torch.where((sum_channel_0 > 0) & (sum_channel_1 > 0), torch.tensor(1).cuda(), labels)
    if torch.rand(1).item() < 0.5:
        labels = torch.where((sum_channel_2 > 0) & (sum_channel_3 > 0), torch.tensor(3).cuda(), labels)
    if torch.rand(1).item() < 0.5:
        labels = torch.where((sum_channel_4 > 0) & (sum_channel_5 > 0), torch.tensor(5).cuda(), labels)

## for onehot-prompt
    label_idx = labels
    labels = torch.zeros(mask_shape[0], 7).cuda().scatter_(1, labels.unsqueeze(1), 1)
    labels = labels.cuda()


# ## for mcmask-transformer
#
#     mask_3_shape = list(mask_t.shape)
#     mask_3_shape[1] = 3
#     mask_3 = torch.zeros(torch.Size(mask_3_shape)).cuda()
#     mask_3[:,0,:,:,:][mask_t.squeeze(1)==2]= 1.0
#     mask_3[:,0,:,:,:][mask_t.squeeze(1)==3]= 1.0
#     mask_3[:,1,:,:,:][mask_t.squeeze(1)==4]= 1.0
#     mask_3[:,1,:,:,:][mask_t.squeeze(1)==5]= 1.0
#     mask_3[:,2,:,:,:][mask_t.squeeze(1)==6]= 1.0
#     mask_3[:,2,:,:,:][mask_t.squeeze(1)==7]= 1.0
#     labels = mask_3
#     labels = labels.cuda()
#     ipdb.set_trace()



# # 初始化标签为 0
#     labels = torch.zeros(mask_shape[0], dtype=torch.long).cuda()
#
#     # 根据条件更新标签
#     labels = torch.where(sum_channel_1 > 0, torch.tensor(2).cuda(), labels)
#     labels = torch.where((sum_channel_1 == 0) & (sum_channel_0 > 0), torch.tensor(1).cuda(), labels)
#     label_idx = labels
#

    # labels = torch.zeros(mask_shape[0], 3).cuda().scatter_(1, labels.unsqueeze(1), 1)
    # labels = labels.cuda()
    # ipdb.set_trace()
    # label_idx = args.label_idx

    loss = tester.ema_model(ct_volume, mask_p, label_idx,labels)
    print("---------------------print {} sample--------------------".format(idx))
    print("label_idx:",label_idx)
    print("sum_channel_0:",sum_channel_0)
    print("sum_channel_1:", sum_channel_1)
    print("sum_channel_2:", sum_channel_2)
    print("sum_channel_3:", sum_channel_3)
    print("sum_channel_4:", sum_channel_4)
    print("sum_channel_5:", sum_channel_5)

    print("noise-loss:",loss)

    # print("before ipdb")
    # ipdb.set_trace()



    with torch.no_grad():


        volume = ct_volume*2.0 - 1.0
        # mask = total_tumor_mask*2.0 - 1.0
        # mask_ = mask_multi[:,0,:,:,:].unsqueeze(1)
        # mask = mask[torch.arange(mask.size(0)), label_idx, :, :, :].unsqueeze(1)


        mask_ =(1.0-mask_p[torch.arange(mask.size(0)),label_idx,:,:,:].unsqueeze(1)).detach()
        # mask_ =(1.0-mask_multi_lesion[:,label_idx,:,:,:].unsqueeze(1)).detach()
        masked_volume = (volume*mask_).detach()

        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        # mask = mask_multi.permute(0,1,-1,-3,-2)
        mask = mask_p[torch.arange(mask.size(0)),label_idx,:,:,:].unsqueeze(1).permute(0,1,-1,-3,-2)
        mask = mask * 2.0 - 1.0
        # ipdb.set_trace()

        #vqgan reconstruction
        recon_loss, x_recon, vq_output, perceptual_loss = vqgan(volume)
        commitment_loss = vq_output['commitment_loss']
        real_recon_loss = F.l1_loss(x_recon, volume)
        print("real_recon_loss_calculate_l1:",real_recon_loss)
        # loss = recon_loss + perceptual_loss + commitment_loss


        recon_volume = x_recon.permute(0,1,-2,-1,-3)
        recon_volume = torch.clamp((recon_volume+1.0)/2.0, min=0.0, max=1.0)

        print("recon_loss:",recon_loss)
        print("perceptual_loss:",perceptual_loss)
        print("commitment_loss:",commitment_loss)
        # print("loss:",loss)






        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                              (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        # recon_volume = vqgan.decode_code(masked_volume_feat)

        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        # ipdb.set_trace()
        # cc = tester.ema_model.conv3d(cc)
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        tester.ema_model.eval()
        sample = tester.ema_model.sample(labels=labels,batch_size=volume.shape[0], cond=cond)
        # ipdb.set_trace()

        diff_loss = F.l1_loss(sample, x_recon)
        print("total_diff_loss:",diff_loss)
        multi_diff_loss_with_recon,masked_sample,masked_recon,count_ = calculate_multiclass_l1_loss(sample, x_recon, mask_multi.permute(0,1,-1,-3,-2))
        print("multi_diff_loss_with_recon:",multi_diff_loss_with_recon)
        multi_diff_loss_with_real,_,masked_real,_ = calculate_multiclass_l1_loss(sample, volume, mask_multi.permute(0,1,-1,-3,-2))
        print("multi_diff_loss_with_real:",multi_diff_loss_with_real)
        multi_diff_loss_recon2real,_,_,_ = calculate_multiclass_l1_loss(x_recon, volume,mask_multi.permute(0,1,-1,-3,-2))
        print("multi_diff_loss_recon2real:",multi_diff_loss_recon2real)
        print("count:",count_)
        masked_sample = masked_sample.permute(0,1,2,-2,-1,-3)
        masked_recon = masked_recon.permute(0,1,2,-2,-1,-3)
        masked_real = masked_real.permute(0,1,2,-2,-1,-3)
        masked_recon = torch.clamp((masked_recon+1.0)/2.0, min=0.0, max=1.0)
        masked_sample = torch.clamp((masked_sample+1.0)/2.0, min=0.0, max=1.0)
        masked_real = torch.clamp((masked_real+1.0)/2.0, min=0.0, max=1.0)



        # ipdb.set_trace()
        #
        # if organ_type == 'liver' or organ_type == 'kidney' :
        #     # post-process
        # mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
        # mask_01 = 1.0 - mask_multi[:,0,:,:,:].unsqueeze(1).permute(0,1,-1,-3,-2)
        mask_01 = mask_p[torch.arange(mask.size(0)),label_idx,:,:,:].unsqueeze(1).permute(0,1,-1,-3,-2)
        # ipdb.set_trace()
        sigma = np.random.uniform(0, 4) # (1, 2)
        mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

        volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
        sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)
        x_recon = torch.clamp((x_recon+1.0)/2.0, min=0.0, max=1.0)

        mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
        final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
        final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        final_volume_with_recon = (1-mask_01_blur)*volume_ +mask_01_blur*x_recon
        final_volume_with_recon = torch.clamp(final_volume_with_recon, min=0.0, max=1.0)
        # elif organ_type == 'pancreas':
        #     final_volume_ = (sample+1.0)/2.0
        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        sample_ = sample_.permute(0,1,-2,-1,-3)
        final_volume_with_recon = final_volume_with_recon.permute(0,1,-2,-1,-3)

        # organ_tumor_mask = organ_mask + total_tumor_mask
        synthetic_tumor_type = "tiny"

    # return final_volume_, organ_tumor_mask
    return final_volume_, mask_t,recon_volume,sample_,final_volume_with_recon,masked_sample,masked_recon,masked_real





# for coronary multi-class-prompt
def synthesize_lesion_with_mask(ct_volume, mask, vqgan, tester,idx,organ_type="coronary"):


    # ipdb.set_trace()


    device=ct_volume.device
    mask_t = mask.clone()
    mask_shape = list(mask_t.shape)  # 将 torch.Size 转换为列表
    mask_shape[1] = 3
    mask_multi = torch.zeros(torch.Size(mask_shape)).cuda()
    mask_shape[1] = 2
    mask_multi_lesion = torch.zeros(torch.Size(mask_shape)).cuda()

    mask_multi[:,1,:,:,:][mask_t.squeeze(1)==3]= 1.0
    mask_multi[:,2,:,:,:][mask_t.squeeze(1)==4]= 1.0
    mask_multi[:,0,:,:,:][mask_t.squeeze(1)==0]= 1.0
    mask_multi[:,0,:,:,:][mask_t.squeeze(1)==1]= 1.0
    mask_multi[:,0,:,:,:][mask_t.squeeze(1)==2]= 1.0
    mask_multi_lesion[:,0,:,:,:][mask_t.squeeze(1)==3]= 1.0
    mask_multi_lesion[:,1,:,:,:][mask_t.squeeze(1)==4]= 1.0

### add prompt label
    mask_shape[1] = 3
    mask_p = torch.zeros(torch.Size(mask_shape)).cuda()
    mask_p[:,1,:,:,:][mask_t.squeeze(1)==3]= 1.0
    mask_p[:,2,:,:,:][mask_t.squeeze(1)==4]= 1.0
    sum_channel_0 = mask_p[:, 1, :, :, :].sum(dim=(1, 2, 3))
    sum_channel_1 = mask_p[:, 2, :, :, :].sum(dim=(1, 2, 3))
    mask_t[mask_t==1] = 0
    mask_t[mask_t==3] = 1
    mask_t[mask_t==4] = 1
    mask_t[mask_t==2] = 1


    # 初始化标签为 0
    labels = torch.zeros(mask_shape[0], dtype=torch.long).cuda()

    # 根据条件更新标签
    labels = torch.where(sum_channel_1 > 0, torch.tensor(2).cuda(), labels)
    labels = torch.where((sum_channel_1 == 0) & (sum_channel_0 > 0), torch.tensor(1).cuda(), labels)
    label_idx = labels

    labels = torch.zeros(mask_shape[0], 3).cuda().scatter_(1, labels.unsqueeze(1), 1)
    labels = labels.cuda()
### for coronary mask generation

    size_types = ['tiny', 'small']
    size_probs = np.array([0.5, 0.5])
    plaque_types = ['calcified', 'noncalcified']
    plaque_probs = np.array([0.5, 0.5])
    total_lesion_mask = []
    organ_mask_np = mask_t.cpu().numpy()
    # ipdb.set_trace()
    for bs in range(organ_mask_np.shape[0]):


        # ipdb.set_trace()
        synthetic_size_type = np.random.choice(size_types, p=size_probs.ravel())
        synthetic_plaque_type = np.random.choice(plaque_types, p=plaque_probs.ravel())

        if synthetic_plaque_type == 'calcified':
            tumor_label = 1
        elif synthetic_plaque_type == 'noncalcified':
            tumor_label = 2
        # ipdb.set_trace()


        labels = (tumor_label*torch.ones(mask.shape[0], dtype=torch.long)).cuda()
        label_idx =labels
        # ipdb.set_trace()
        labels = torch.zeros((mask.shape[0], 3)).cuda().scatter_(1, labels.unsqueeze(1), 1)
        labels = labels.cuda()

        # ipdb.set_trace()
        tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_size_type, "coronary")
        total_lesion_mask.append(torch.from_numpy(tumor_mask)[None,:])
    total_tumor_mask = torch.stack(total_lesion_mask, dim=0).to(dtype=torch.float32, device=device)
    # ipdb.set_trace()
    total_mask = total_tumor_mask*label_idx[0] + mask_t*2
    print("unique values of total_mask:", torch.unique(total_mask))


    organ_mask_np= organ_mask_np*2
    # mask_p = torch.zeros(torch.Size(mask_shape)).cuda()
    #
    # mask_p[:,label_idx,:,:,:][mask_t.squeeze(1)==3]= 1.0

    # ipdb.set_trace()
    # label_idx = args.label_idx

    loss = tester.ema_model(ct_volume, mask_p, label_idx,labels)
    print("---------------------print {} sample--------------------".format(idx))
    print("label_idx:",label_idx)
    print("sum_channel_0:",sum_channel_0)
    print("sum_channel_1:", sum_channel_1)
    print("noise-loss:",loss)

    # print("before ipdb")
    # ipdb.set_trace()



    with torch.no_grad():


        volume = ct_volume*2.0 - 1.0
        # mask = total_tumor_mask*2.0 - 1.0
        # mask_ = mask_multi[:,0,:,:,:].unsqueeze(1)
        # mask = mask[torch.arange(mask.size(0)), label_idx, :, :, :].unsqueeze(1)


        # mask_ =(1.0-mask_p[torch.arange(mask.size(0)),label_idx,:,:,:].unsqueeze(1)).detach()
        mask_ = (1.0 -total_tumor_mask).detach()
        # mask_ =(1.0-mask_multi_lesion[:,label_idx,:,:,:].unsqueeze(1)).detach()
        masked_volume = (volume*mask_).detach()

        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        # mask = mask_multi.permute(0,1,-1,-3,-2)
        # mask = mask_p[torch.arange(mask.size(0)),label_idx,:,:,:].unsqueeze(1).permute(0,1,-1,-3,-2)
        mask = total_tumor_mask.permute(0,1,-1,-3,-2)
        print("unique values of mask:", torch.unique(mask))
        mask = mask * 2.0 - 1.0
        # ipdb.set_trace()

        #vqgan reconstruction
        recon_loss, x_recon, vq_output, perceptual_loss = vqgan(volume)
        commitment_loss = vq_output['commitment_loss']
        real_recon_loss = F.l1_loss(x_recon, volume)
        print("real_recon_loss_calculate_l1:",real_recon_loss)
        # loss = recon_loss + perceptual_loss + commitment_loss


        recon_volume = x_recon.permute(0,1,-2,-1,-3)
        recon_volume = torch.clamp((recon_volume+1.0)/2.0, min=0.0, max=1.0)

        print("recon_loss:",recon_loss)
        print("perceptual_loss:",perceptual_loss)
        print("commitment_loss:",commitment_loss)
        # print("loss:",loss)






        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                              (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        # recon_volume = vqgan.decode_code(masked_volume_feat)

        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        # ipdb.set_trace()
        # cc = tester.ema_model.conv3d(cc)
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        tester.ema_model.eval()
        print("labels:",labels)
        sample = tester.ema_model.sample(labels=labels,batch_size=volume.shape[0], cond=cond)
        # ipdb.set_trace()

        diff_loss = F.l1_loss(sample, x_recon)
        print("total_diff_loss:",diff_loss)
        multi_diff_loss_with_recon,masked_sample,masked_recon,count_ = calculate_multiclass_l1_loss(sample, x_recon, mask_multi.permute(0,1,-1,-3,-2))
        print("multi_diff_loss_with_recon:",multi_diff_loss_with_recon)
        multi_diff_loss_with_real,_,masked_real,_ = calculate_multiclass_l1_loss(sample, volume, mask_multi.permute(0,1,-1,-3,-2))
        print("multi_diff_loss_with_real:",multi_diff_loss_with_real)
        multi_diff_loss_recon2real,_,_,_ = calculate_multiclass_l1_loss(x_recon, volume,mask_multi.permute(0,1,-1,-3,-2))
        print("multi_diff_loss_recon2real:",multi_diff_loss_recon2real)
        print("count:",count_)
        masked_sample = masked_sample.permute(0,1,2,-2,-1,-3)
        masked_recon = masked_recon.permute(0,1,2,-2,-1,-3)
        masked_real = masked_real.permute(0,1,2,-2,-1,-3)
        masked_recon = torch.clamp((masked_recon+1.0)/2.0, min=0.0, max=1.0)
        masked_sample = torch.clamp((masked_sample+1.0)/2.0, min=0.0, max=1.0)
        masked_real = torch.clamp((masked_real+1.0)/2.0, min=0.0, max=1.0)



        # ipdb.set_trace()
        #
        # if organ_type == 'liver' or organ_type == 'kidney' :
        #     # post-process
        # mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
        # mask_01 = 1.0 - mask_multi[:,0,:,:,:].unsqueeze(1).permute(0,1,-1,-3,-2)
        # mask_01 = mask_p[torch.arange(mask.size(0)),label_idx,:,:,:].unsqueeze(1).permute(0,1,-1,-3,-2)
        mask_01 = total_tumor_mask.permute(0,1,-1,-3,-2)
        # ipdb.set_trace()
        sigma = np.random.uniform(0, 4) # (1, 2)
        mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

        volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
        sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)
        x_recon = torch.clamp((x_recon+1.0)/2.0, min=0.0, max=1.0)

        mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
        final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
        final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        final_volume_with_recon = (1-mask_01_blur)*volume_ +mask_01_blur*x_recon
        final_volume_with_recon = torch.clamp(final_volume_with_recon, min=0.0, max=1.0)
        # elif organ_type == 'pancreas':
        #     final_volume_ = (sample+1.0)/2.0
        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        sample_ = sample_.permute(0,1,-2,-1,-3)
        final_volume_with_recon = final_volume_with_recon.permute(0,1,-2,-1,-3)

        # organ_tumor_mask = organ_mask + total_tumor_mask

    # return final_volume_, organ_tumor_mask
    return final_volume_, total_mask,organ_mask_np,recon_volume,sample_,final_volume_with_recon,masked_sample,masked_recon,masked_real

# for generate  tumor using alltumor prompt
def synthesize_tumor_in_alltumor(ct_volume, organ_mask, organ_type, vqgan, tester):
    device=ct_volume.device
    tumor_label = 0
    # label process
    # generate tumor mask
    tumor_types = ['tiny', 'small','medium', 'large']
    tumor_probs = np.array([0.25, 0.25,0.25,0.25])
    total_tumor_mask = []
    organ_mask_np = organ_mask.cpu().numpy()




    with torch.no_grad():
        # get model input
        for bs in range(organ_mask_np.shape[0]):
            synthetic_tumor_type = np.random.choice(tumor_types, p=tumor_probs.ravel())
            if organ_type == 'kidney':
                if synthetic_tumor_type == 'tiny' or synthetic_tumor_type == 'small':
                    tumor_label = 1
                elif synthetic_tumor_type == 'medium' or synthetic_tumor_type == 'large':
                    tumor_label = 2
            elif organ_type == 'liver':
                if synthetic_tumor_type == 'tiny' or synthetic_tumor_type == 'small':
                    tumor_label = 3
                elif synthetic_tumor_type == 'medium' or synthetic_tumor_type == 'large':
                    tumor_label = 4

            elif organ_type == 'pancreas':
                if synthetic_tumor_type == 'tiny' or synthetic_tumor_type == 'small':
                    tumor_label = 5
                elif synthetic_tumor_type == 'medium' or synthetic_tumor_type == 'large':
                    tumor_label = 6
        # 对labels进行 tumor_label 的赋值
        #     ipdb.set_trace()

            labels = (tumor_label*torch.ones(organ_mask.shape[0], dtype=torch.long)).cuda()
            label_idx =labels
            labels = torch.zeros((organ_mask.shape[0], 7)).cuda().scatter_(1, labels.unsqueeze(1), 1)
            labels = labels.cuda()

            # ipdb.set_trace()
            tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
            total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)
        import ipdb
        mask_shape = list(total_tumor_mask.shape)  # 将 torch.Size 转换为列表
        # mask_shape[1] = 7
        # mask_ori = torch.zeros(torch.Size(mask_shape)).cuda()
        # mask_ori[:,int(label_idx),:,:,:][total_tumor_mask.squeeze(1)==1]= 1.0

        ## for mcmask-transformer
        mask_shape[1] = 3
        mask_ori = torch.zeros(torch.Size(mask_shape)).cuda()
        mask_ori[:,(int(label_idx)-1)//2,:,:,:][total_tumor_mask.squeeze(1)==1]= 1.0
        print("mask_ori shape: ", mask_ori.shape, "label_idx: ", label_idx,"synthetic_tumor_type: ", synthetic_tumor_type)

        # mask_ori = total_tumor_mask.clone()
        # ipdb.set_trace()

        volume = ct_volume*2.0 - 1.0
        mask = total_tumor_mask*2.0 - 1.0
        mask_ = 1-total_tumor_mask
        masked_volume = (volume*mask_).detach()

        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        mask = mask.permute(0,1,-1,-3,-2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                              (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0

        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        tester.ema_model.eval()
        sample = tester.ema_model.sample(labels=mask_ori,batch_size=volume.shape[0], cond=cond)
        # sample = tester.ema_model.sample(labels=labels,batch_size=volume.shape[0], cond=cond)

        if organ_type == 'liver' or organ_type == 'kidney' :
            # post-process
            mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
            sigma = np.random.uniform(0, 4) # (1, 2)
            mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

            volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
            sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

            mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
            final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
            final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        elif organ_type == 'pancreas':
            final_volume_ = (sample+1.0)/2.0
        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        organ_tumor_mask = organ_mask + total_tumor_mask
    # ipdb.set_trace()

    return final_volume_, organ_tumor_mask,synthetic_tumor_type

# generate tumor using mcmask-input
def synthesize_tumor_in_alltumor_mcmaskinput(ct_volume, organ_mask, organ_type, vqgan, tester):
    device=ct_volume.device
    tumor_label = 0
    # label process
    # generate tumor mask
    tumor_types = ['tiny', 'small','medium', 'large']
    tumor_probs = np.array([0.25, 0.25,0.25,0.25])
    total_tumor_mask = []
    organ_mask_np = organ_mask.cpu().numpy()




    with torch.no_grad():
        # get model input
        for bs in range(organ_mask_np.shape[0]):
            synthetic_tumor_type = np.random.choice(tumor_types, p=tumor_probs.ravel())
            if organ_type == 'kidney':
                if synthetic_tumor_type == 'tiny' or synthetic_tumor_type == 'small':
                    tumor_label = 1
                elif synthetic_tumor_type == 'medium' or synthetic_tumor_type == 'large':
                    tumor_label = 2
            elif organ_type == 'liver':
                if synthetic_tumor_type == 'tiny' or synthetic_tumor_type == 'small':
                    tumor_label = 3
                elif synthetic_tumor_type == 'medium' or synthetic_tumor_type == 'large':
                    tumor_label = 4

            elif organ_type == 'pancreas':
                if synthetic_tumor_type == 'tiny' or synthetic_tumor_type == 'small':
                    tumor_label = 5
                elif synthetic_tumor_type == 'medium' or synthetic_tumor_type == 'large':
                    tumor_label = 6
            # 对labels进行 tumor_label 的赋值
            #     ipdb.set_trace()

            labels = (tumor_label*torch.ones(organ_mask.shape[0], dtype=torch.long)).cuda()
            label_idx =labels
            labels = torch.zeros((organ_mask.shape[0], 7)).cuda().scatter_(1, labels.unsqueeze(1), 1)
            labels = labels.cuda()

            # ipdb.set_trace()
            tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
            total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)
        import ipdb
        mask_shape = list(total_tumor_mask.shape)  # 将 torch.Size 转换为列表
        # mask_shape[1] = 7
        # mask_ori = torch.zeros(torch.Size(mask_shape)).cuda()
        # mask_ori[:,int(label_idx),:,:,:][total_tumor_mask.squeeze(1)==1]= 1.0

        ## for mcmask-input
        mask_shape[1] = 7
        mask_ori = torch.zeros(torch.Size(mask_shape)).cuda()
        mask_ori[:,(int(label_idx)),:,:,:][total_tumor_mask.squeeze(1)==1]= 1.0
        print("mask_ori shape: ", mask_ori.shape, "label_idx: ", label_idx,"synthetic_tumor_type: ", synthetic_tumor_type)

        # mask_ori = total_tumor_mask.clone()
        # ipdb.set_trace()

        volume = ct_volume*2.0 - 1.0
        mask = total_tumor_mask*2.0 - 1.0
        mask_ = 1-total_tumor_mask
        masked_volume = (volume*mask_).detach()

        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        mask = mask.permute(0,1,-1,-3,-2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                              (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0

        cc = torch.nn.functional.interpolate(mask_ori, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)
        print("cond shape: ", cond.shape, "cc shape:",cc.shape, "mask_ori shape: ", mask_ori.shape)

        # diffusion inference and decoder
        tester.ema_model.eval()
        sample = tester.ema_model.sample(labels=None,batch_size=volume.shape[0], cond=cond)
        # sample = tester.ema_model.sample(labels=labels,batch_size=volume.shape[0], cond=cond)

        if organ_type == 'liver' or organ_type == 'kidney' :
            # post-process
            mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
            sigma = np.random.uniform(0, 4) # (1, 2)
            mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

            volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
            sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

            mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
            final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
            final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        elif organ_type == 'pancreas':
            final_volume_ = (sample+1.0)/2.0
        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        organ_tumor_mask = organ_mask + total_tumor_mask
    # ipdb.set_trace()

    return final_volume_, organ_tumor_mask,synthetic_tumor_type

def synthesize_early_tumor(ct_volume, organ_mask, organ_type, vqgan, tester):
    device=ct_volume.device

    # generate tumor mask
    tumor_types = ['tiny', 'small']
    tumor_probs = np.array([0.5, 0.5])
    total_tumor_mask = []
    organ_mask_np = organ_mask.cpu().numpy()
    with torch.no_grad():
        # get model input
        for bs in range(organ_mask_np.shape[0]):
            synthetic_tumor_type = np.random.choice(tumor_types, p=tumor_probs.ravel())
            tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
            total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)

        volume = ct_volume*2.0 - 1.0
        mask = total_tumor_mask*2.0 - 1.0
        mask_ = 1-total_tumor_mask
        masked_volume = (volume*mask_).detach()
        
        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        mask = mask.permute(0,1,-1,-3,-2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        
        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        tester.ema_model.eval()
        sample = tester.ema_model.sample(batch_size=volume.shape[0], cond=cond)

        if organ_type == 'liver' or organ_type == 'kidney' :
            # post-process
            mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
            sigma = np.random.uniform(0, 4) # (1, 2)
            mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

            volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
            sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

            mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
            final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
            final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        elif organ_type == 'pancreas':
            final_volume_ = (sample+1.0)/2.0
        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        organ_tumor_mask = organ_mask + total_tumor_mask

    return final_volume_, organ_tumor_mask


def synthesize_medium_tumor(ct_volume, organ_mask, organ_type, vqgan, sampler, ddim_ts=50):
    device=ct_volume.device

    total_tumor_mask = []
    organ_mask_np = organ_mask.cpu().numpy()
    with torch.no_grad():
        # get model input
        for bs in range(organ_mask_np.shape[0]):
            synthetic_tumor_type = 'medium'
            tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
            total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)

        volume = ct_volume*2.0 - 1.0
        mask = total_tumor_mask*2.0 - 1.0
        mask_ = 1-total_tumor_mask
        masked_volume = (volume*mask_).detach()
        
        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        mask = mask.permute(0,1,-1,-3,-2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        
        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        shape = masked_volume_feat.shape[-4:]
        samples_ddim, _ = sampler.sample(S=ddim_ts,
                                        conditioning=cond,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False)
        samples_ddim = (((samples_ddim + 1.0) / 2.0) * (vqgan.codebook.embeddings.max() -
                                                        vqgan.codebook.embeddings.min())) + vqgan.codebook.embeddings.min()

        sample = vqgan.decode(samples_ddim, quantize=True)
        
        if organ_type == 'liver' or organ_type == 'kidney':
            # post-process
            mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
            sigma = np.random.uniform(0, 4) # (1, 2)
            mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

            volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
            sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

            mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
            final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
            final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        elif organ_type == 'pancreas':
            final_volume_ = (sample+1.0)/2.0

        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        organ_tumor_mask = torch.zeros_like(organ_mask)
        organ_tumor_mask[organ_mask==1] = 1
        organ_tumor_mask[total_tumor_mask==1] = 2

    return final_volume_, organ_tumor_mask

def synthesize_large_tumor(ct_volume, organ_mask, organ_type, vqgan, sampler, ddim_ts=50):
    device=ct_volume.device

    total_tumor_mask = []
    organ_mask_np = organ_mask.cpu().numpy()
    with torch.no_grad():
        # get model input
        for bs in range(organ_mask_np.shape[0]):
            synthetic_tumor_type = 'large'
            tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
            total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)

        volume = ct_volume*2.0 - 1.0
        mask = total_tumor_mask*2.0 - 1.0
        mask_ = 1-total_tumor_mask
        masked_volume = (volume*mask_).detach()
        
        volume = volume.permute(0,1,-1,-3,-2)
        masked_volume = masked_volume.permute(0,1,-1,-3,-2)
        mask = mask.permute(0,1,-1,-3,-2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        
        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        # diffusion inference and decoder
        shape = masked_volume_feat.shape[-4:]
        samples_ddim, _ = sampler.sample(S=ddim_ts,
                                        conditioning=cond,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False)
        samples_ddim = (((samples_ddim + 1.0) / 2.0) * (vqgan.codebook.embeddings.max() -
                                                        vqgan.codebook.embeddings.min())) + vqgan.codebook.embeddings.min()

        sample = vqgan.decode(samples_ddim, quantize=True)

        if organ_type == 'liver' or organ_type == 'kidney':
            # post-process
            mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
            sigma = np.random.uniform(0, 4) # (1, 2)
            mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy()*1.0, sigma=[0,0,sigma,sigma,sigma])

            volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
            sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

            mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
            final_volume_ = (1-mask_01_blur)*volume_ +mask_01_blur*sample_
            final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
        elif organ_type == 'pancreas':
            final_volume_ = (sample+1.0)/2.0
            
        final_volume_ = final_volume_.permute(0,1,-2,-1,-3)
        organ_tumor_mask = torch.zeros_like(organ_mask)
        organ_tumor_mask[organ_mask==1] = 1
        organ_tumor_mask[total_tumor_mask==1] = 2

    return final_volume_, organ_tumor_mask