import os
import sys
sys.path.append(os.getcwd())
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from vq_gan_3d.model import VQGAN
from callbacks import ImageLogger, VideoLogger
import hydra
from omegaconf import DictConfig, open_dict
from dataset.dataloader import get_loader
import argparse
import logging
logging.disable(logging.WARNING)
import ipdb
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total', total_num/(1024*1024.0), 'Trainable', trainable_num/(1024*1024.0))
    return {'Total': total_num, 'Trainable': trainable_num}


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig, args=None):
    pl.seed_everything(cfg.model.seed)
    # ipdb.set_trace()

    # ## old
    train_dataloader, _, _, = get_loader(cfg.dataset)

    val_dataloader=None

    # automatically adjust learning rate
    base_lr = cfg.model.lr

    with open_dict(cfg):
        cfg.model.lr = 1 * (1/8.) * (2/4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)

    model = VQGAN(cfg)

    get_parameter_number(model)
    save_step = 500
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=save_step,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=1000, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # load the most recent checkpoint file
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        if cfg.model.resume:
            log_folder = 'version_'+str(cfg.model.resume_version)
            if len(log_folder) > 0:
                ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            print('resume training:', cfg.model.resume_from_checkpoint)
        else:
            log_folder = ckpt_file = ''
            version_id_used = step_used = 0
            for folder in os.listdir(base_dir):
                version_id = int(folder.split('_')[1])
                if version_id > version_id_used:
                    version_id_used = version_id
                    log_folder = 'version_'+str(version_id_used+1)
            if len(log_folder) > 0:
                ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')

    if not cfg.model.pretrained_checkpoint is None:
        print("cfg.model.pretrained_checkpoint:", cfg.model.pretrained_checkpoint)
        model.load_from_checkpoint(cfg.model.pretrained_checkpoint)
        model=type(model).load_from_checkpoint(cfg.model.pretrained_checkpoint)
        # print('load pretrained model:',cfg.model.pretrained_checkpoint )
        # model.load_from_checkpoint()
        print('load pretrained model:', cfg.model.pretrained_checkpoint)
    else:
        print('No pretrained model loaded, training from scratch.')


    ## start test_vqgan
    # model = model.cuda()
    # model_state_dict_before = model.state_dict()




    model = model.cuda()

    model.eval()
    #
    for idx, batch_data in enumerate(train_dataloader):

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target, data_names = batch_data['image'], batch_data['label'], batch_data['name']
        data, target = data.cuda(), target.cuda()

        with torch.no_grad():


            # volume = data*2.0 - 1.0
            # mask = total_tumor_mask*2.0 - 1.

            volume = data.permute(0,1,-1,-3,-2)

            # mask = mask * 2.0 - 1.0
            # ipdb.set_trace()

            #vqgan reconstruction
            recon_loss, x_recon, vq_output, perceptual_loss = model(volume)
            commitment_loss = vq_output['commitment_loss']
            # loss = recon_loss + perceptual_loss + commitment_loss


            recon_volume = x_recon.permute(0,1,-2,-1,-3)
            volume = volume.permute(0,1,-2,-1,-3)
            print("recon_loss:",recon_loss)
            print("perceptual_loss:",perceptual_loss)
            print("commitment_loss:",commitment_loss)
            # import nibabel as nib
            # import numpy as np
            # nii_img = nib.Nifti1Image(recon_volume[1][0].detach().cpu().numpy().astype(np.float32), affine=np.eye(4))
            # nii_img.to_filename(f"synth_data/{idx}_img_recon.nii.gz")
            # nii_img = nib.Nifti1Image(volume[1][0].detach().cpu().numpy().astype(np.float32), affine=np.eye(4))
            # nii_img.to_filename(f"synth_data/{idx}_img.nii.gz")
    # #end test_vqgan







    accelerator = None
    if cfg.model.gpus > 1:
        accelerator = 'ddp'






    # ipdb.set_trace()

    trainer = pl.Trainer(
        gpus=cfg.model.gpus,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        resume_from_checkpoint=cfg.model.pretrained_checkpoint,
        # resume_from_checkpoint="/home/mil/ruan/FreeLesion/STEP1.AutoencoderModel/checkpoints/vq_gan/synt/coronary_part_96_1000epoch/lightning_logs/version_1/checkpoints/epoch=154-step=10999-10000-train/recon_loss=0.02.ckpt",
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        # max_steps=10,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        accelerator=accelerator,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.validate(model, train_dataloader)
    print('training finished')
    # ipdb.set_trace()


if __name__ == '__main__':
    run()
