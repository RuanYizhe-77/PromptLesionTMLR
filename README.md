
## 0. Installation



See [installation instructions](documents/INSTALL.md) to create an environment and obtain requirements.

## 1. Train Autoencoder Model
You can train Autoencoder Model on your own CCTA dataset. ( change a_min to 0.0 & a_max to 255.0 in config/dataset/synt_ct.yaml)

You can also train Autoencoder Model on AbdomenAtlas 1.0 dataset by your own for tumor  ( change a_min to -175 & a_max to 250.0 in config/dataset/synt_ct.yaml). The release of AbdomenAtlas 1.0 can be found at [https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas_1.0_Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas_1.0_Mini).
```bash
cd STEP1.AutoencoderModel
datapath= (e.g., /data/bdomenAtlasMini1.0/)
gpu_num=1
cache_rate=0.01
batch_size=4
dataset_list="coronary_part"
pretrain_path='your/pretrian/checkpoint'
python train.py dataset.data_root_path=$datapath dataset.dataset_list=$dataset_list dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size model.gpus=$gpu_num model.pretrained_checkpoint="'$pretrain_path'"
```



## 2. Train PromptLesion Model


Download the public dataset MSD-Liver (More datasets can be seen in [installation instructions](documents/INSTALL.md)). 
```bash
wget https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/Task03_Liver.tar.gz
tar -zxvf Task03_Liver.tar.gz
```
Preprocessed labels for early-stage tumors and mid-/late- stage tumors.
```bash
wget https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/preprocessed_labels.tar.gz
tar -zxvf preprocessed_labels.tar.gz
```
<details>
<summary style="margin-left: 25px;">Preprocess details</summary>
<div style="margin-left: 25px;">

1. Download the dataset according to the [installation instructions](documents/INSTALL.md).  
2. Modify `data_dir` and `tumor_save_dir` in [data_transfer.py](https://github.com/MrGiovanni/DiffTumor/blob/main/data_transfer.py).
3. `python -W ignore data_transfer.py`
</div>
</details>
Start training.

```bash
## for coronary lesion
cd STEP2.PromptLesion/

vqgan_ckpt=<pretrained-AutoencoderModel> (e.g., /pretrained_models/AutoencoderModel.ckpt)
fold=0
cache_rate=0.05
datapath= (e.g., /data/CCTA/)

python train.py dataset.name=coronary_lesion_train dataset.fold=$fold dataset.data_root_path=$datapath  dataset.dataset_list=['coronary_data_fold'] dataset.uniform_sample=False model.results_folder_postfix="coronary_data_fold$fold"  model.vqgan_ckpt=$vqgan_ckpt dataset.cache_rate=$cache_rate

```
```bash
## for tumor 
replace (def train) in line 1795 in diffusion.py with the code from line 1596-1790
cd STEP2.PromptLesion/
vqgan_ckpt=<pretrained-AutoencoderModel> (e.g., /pretrained_models/AutoencoderModel.ckpt)
fold=0
cache_rate=0.05
python train.py dataset.name=alltumor_lesion_train dataset.fold=$fold   dataset.dataset_list=['alltumor_data_fold'] dataset.uniform_sample=False model.results_folder_postfix="alltumor_data_fold$fold"  model.vqgan_ckpt=$vqgan_ckpt dataset.cache_rate=$cache_rate


```


## 3. Train Segmentation Model

Download healthy CT data

<details>
<summary style="margin-left: 25px;">from Huggingface</summary>
<div style="margin-left: 25px;">

(More details can be seen in the corresponding [huggingface repository](https://huggingface.co/datasets/qicq1c/HealthyCT)).
```bash
mkdir HealthyCT
cd HealthyCT
huggingface-cli download qicq1c/HealthyCT  --repo-type dataset --local-dir .  --cache-dir ./cache
cat healthy_ct.zip* > HealthyCT.zip
rm -rf healthy_ct.zip* cache
unzip -o -q HealthyCT.zip -d /HealthyCT
```
</div>
</details>

Prepare Autoencoder and Diffusion Model. Put the pre-trained weights to `STEP3.SegmentationModel/TumorGeneration/model_weight`

Start training.
```bash
For tumor segmentation
cd STEP3.SegmentationModel

healthy_datapath=your/healthy/ct       (/HealthyCT/healthy_ct/)
datapath=/data
cache_rate=0.05
batch_size=12
val_every=50
workers=4
organ=liver
fold=0
backbone=unet #(uunet for uunet backbone)
type=mix
structure=prompt
logdir="runs/$type.$structure.$organ.fold$fold.$backbone"
datafold_dir=cross_eval/"$organ"_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed=False --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir  --is_save_synth

For Coronary segmentation

cd STEP3.SegmentationModel
# sythetic data generation
# make sure you have the pre-trained Autoencoder & Diffusion model in /pretrained_models/AutoencoderModel.ckpt & rewrite the indicator in  synt_model_prepare in utils.py
healthy_datapath=your/healthy/CCTA    
synth_dir = your/synth/ct  
cache_rate=0.05
batch_size=12
val_every=50
workers=4
organ=coronary
fold=0
datafold_dir=cross_eval/"$organ"_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
python -W ignore lesionGen.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers  --batch_size=$batch_size --save_checkpoint --distributed=False --noamp --organ_type $organ --organ_model $organ --tumor_type lesion --fold $fold --ddim_ts 50  --healthy_data_root $healthy_datapath --synth_dir $synth_dir --datafold_dir $datafold_dir  --is_save_synth

#train segmentation model
healthy_datapath=your/healthy/ct       (/HealthyCT/healthy_ct/)
datapath=/data
synth_dir = your/synth/ct 
cache_rate=0.05
batch_size=12
val_every=50
workers=4
organ=coronary
fold=0
backbone=unet #(uunet for uunet backbone)
type=mix
structure=prompt
logdir="runs/$type.$structure.$organ.fold$fold.$backbone"
datafold_dir=cross_eval/"$organ"_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --synth_dir $synth_dir --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed=False --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir  


```



## 4. Evaluation

```bash
cd SegmentationModel
datapath=/data
organ=liver
fold=0
datafold_dir=cross_eval/"$organ"_aug_data_fold/

# U-Net
python -W ignore validation.py --model=unet --data_root $datapath --datafold_dir $datafold_dir --tumor_type tumor --organ_type $organ --fold $fold --log_dir runs/saved_model --save_dir out/$organ/mix$organ.fold$fold.unet


```
## CCTA Preprocessing

```bash
## We provide a script to convert CCTA tiff files to NIfTI format.
python tif2nifti.py ## change your input dir and output dir in tif2nifti.py
```
## Pre-trained Models
### CCTA Models
### 1.Usage Notes for CCTA Pretrained Weights

We provide the pretrained model weights of Autoencoder model and Promplesion model trained on our private CCTA dataset, you can download them from the following links: https://drive.google.com/drive/folders/1mlQ7BD9Gchedz5kEHM4Lz36LNX-ZhkL8?usp=drive_link

### 2.Important Recommendation:
Due to the challenge of domain shift in medical imaging, the performance of these weights on your own dataset may vary. For optimal results, we strongly recommend: Use our public scripts to train a new model on your dataset.
### CT Models
We provide the pretrained model weights of Autoencoder model and Promplesion model trained on public Kidney, Liver and Pancreas dataset, you can download them from the following links: https://drive.google.com/drive/folders/1GIxvmlGJ5N9NiP21-mt7I_SGke3i17hf?usp=drive_link

## Acknowledgement
This codebase is built upon the excellent NVIDIA MONAI framework. We also thank the authors of DiffTumor for their public codebase, which served as a valuable reference for our implementation.