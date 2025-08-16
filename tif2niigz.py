
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import nibabel as nib
import ipdb

class Gray2Mask(object):
    """ convert gray-scale image to 2D mask
        red - 76 : low-density plaque --> 4
        black - 0 : background --> 0
        orange - 151 : calcification --> 3
        white - 255 : Border of the artery (small in healthy patients) --> 2
        blue - 29 : inside of the artery --> 1
    """
    def __call__(self, gray):


        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[gray == 76] = 4
        mask[gray == 151] = 3
        mask[gray == 255] = 2
        mask[gray == 29] = 1

        return  mask
class HU2Gray(object):
    def __init__(self, hu_max=1440.0, hu_min=-1024.0):
        self.hu_max = hu_max
        self.hu_min = hu_min
        self.scale = float(255) / (self.hu_max - self.hu_min)

    def __call__(self, image):
        """ convert HU value to gray scale [0, 255]
        hu: numpy ndarray, Image of HU value, [H, W]

        """
        # print("sample:",sample)
        # print("sample_len:", len(sample))
        # image, mask,mask_seg,mask_dis= sample
        # print(mask_dis)
        # print("mask_dis_shape:", mask_dis.shape)
        image =  (image - self.hu_min) * self.scale

        return image

def validate_tif_file(tif_file):
    """检查 .tif 文件是否损坏"""
    try:
        img = Image.open(tif_file)
        img.verify()  # 验证图像文件是否完整
        return True
    except (UnidentifiedImageError, IOError):
        return False
def check_label_positivity(label):
    """
    检查标签数组中是否包含值 3 或 4。

    参数:
        label (np.array): 3D 标签数组。

    返回:
        str: "positive" 如果数组包含 3 或 4，否则返回 "negative"。
    """
    # 使用 numpy 的布尔索引来检查数组中是否存在 3 或 4
    if np.any((label == 3) | (label == 4)):
        return True
    else:
        return False


def tif_to_nii(input_dir, output_dir):
    """
    将每个患者目录中的 .tif 图像和对应标签文件转换为 .nii.gz 文件，并按 patient_name + artery_index 命名的文件夹存储。

    参数:
        image_dir (str): 输入图像文件夹路径，包含 patient_name/artery_index/*.tif 结构的图像文件。
        label_dir (str): 输入标签文件夹路径，包含与图像文件夹结构相同的标签 .tif 文件。
        output_dir (str): 输出文件夹的路径，将保存 .nii.gz 文件。
    """

    # 遍历所有患者文件夹
    for patient_name in os.listdir(input_dir):
        patient_path = os.path.join(input_dir, patient_name)



        if os.path.isdir(patient_path):
            for artery_index in os.listdir(patient_path):
                artery_path = os.path.join(patient_path, artery_index)
                artery_image_path = os.path.join(artery_path, 'applicate/image')
                artery_label_path = os.path.join(artery_path, 'applicate/mask')






                if os.path.isdir(artery_image_path) and os.path.isdir(artery_label_path):
                    # 收集所有图像和标签 .tif 文件路径
                    image_tif_files = sorted([os.path.join(artery_image_path, f) for f in os.listdir(artery_image_path) if f.endswith('.tiff')])
                    label_tif_files = sorted([os.path.join(artery_label_path, f) for f in os.listdir(artery_label_path) if f.endswith('.tiff')])

                    # 确保图像和标签文件数量一致
                    if len(image_tif_files) != len(label_tif_files):
                        print(f"Warning: Mismatch in number of files for {artery_image_path} and {artery_label_path}. Skipping.")
                        continue

                    valid_image_tif_files = []
                    valid_label_tif_files = []

                    for img_file, lbl_file in zip(image_tif_files, label_tif_files):
                        if validate_tif_file(img_file) and validate_tif_file(lbl_file):
                            valid_image_tif_files.append(img_file)
                            valid_label_tif_files.append(lbl_file)
                        else:
                            print(f"Warning: Either image {img_file} or label {lbl_file} is corrupted. Skipping both.")

                    # 确保有有效的 .tif 文件存在
                    if len(valid_image_tif_files) > 0:
                        # 读取第一个 .tif 文件以获取图像尺寸
                        first_img = Image.open(valid_image_tif_files[0])
                        img_shape = (len(valid_image_tif_files),) + first_img.size[::-1]  # (切片数量, 高度, 宽度)

                        # 创建一个空的 numpy 数组来存储图像数据 (float64 类型)
                        img_array = np.zeros(img_shape, dtype=np.float64)
                        lbl_array = np.zeros(img_shape, dtype=np.uint8)

                        # 读取每个 .tif 文件并存入数组中
                        for i, (img_file, lbl_file) in enumerate(zip(valid_image_tif_files, valid_label_tif_files)):
                            img_array[i, :, :] = np.array(Image.open(img_file)).astype(np.float64)
                            lbl_array[i, :, :] = np.array(Image.open(lbl_file)).astype(np.uint8)

                        img_array = HU2Gray()(img_array)
                        lbl_array = Gray2Mask()(lbl_array)

                        label_unique = np.unique(lbl_array)
                        print("label_unique:", label_unique)
                        if 3 not in label_unique and 4 not in label_unique:
                            print(f"Warning: No label 3 or 4 found in {artery_label_path}. Skipping.")
                            continue

                        # 创建 NIfTI 图像
                        nii_img = nib.Nifti1Image(img_array, affine=np.eye(4))
                        nii_lbl = nib.Nifti1Image(lbl_array, affine=np.eye(4))

                        # 构造新文件夹路径
                        new_folder_name = f"{patient_name}_{artery_index}"
                        new_folder_path = os.path.join(output_dir, new_folder_name)
                        os.makedirs(new_folder_path, exist_ok=True)

                        # 保存 CT 文件
                        ct_filepath = os.path.join(new_folder_path, 'ct.nii.gz')
                        nib.save(nii_img, ct_filepath)

                        # 保存标签文件到新建的 segmentation 文件夹
                        segmentation_folder = os.path.join(new_folder_path, 'segmentations')
                        os.makedirs(segmentation_folder, exist_ok=True)
                        plaque_filepath = os.path.join(segmentation_folder, 'plaque.nii.gz')
                        nib.save(nii_lbl, plaque_filepath)

                        print(f"Saved CT: {ct_filepath}")
                        print(f"ct shape: {img_array.shape}")
                        print(f"Saved Label: {plaque_filepath}")
                        print(f"label shape: {lbl_array.shape}")
                    else:
                        print(f"No valid .tif files found for {artery_image_path} and {artery_label_path}, skipping this artery.")
                    # ipdb.set_trace()


if __name__ == "__main__":
    # 输入和输出路径
    input_dir = "/data/utsubo0/users/ruan/CPR_multiview_interp2_huang_full/"  # 替换为图像文件夹路径

    # output_dir = "/data/utsubo0/users/ruan/CPR_multiview_niigz_gray"# 替换为输出文件夹路径
    output_dir = "/data/utsubo0/users/ruan/CPR_multiview_niigz_gray_unhealthy"
    # output_dir = "/data/utsubo0/users/ruan/CPR_multiview_niigz"

    os.makedirs(output_dir, exist_ok=True)

    # 转换图像文件
    tif_to_nii(input_dir, output_dir)
