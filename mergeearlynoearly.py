import os
import nibabel as nib
import numpy as np

# 定义器官和标签目录
organs = ['kidney', 'liver', 'pancreas']
base_dir = '/data/utsubo0/users/ruan/preprocessed_labels/'  # 设定基础目录
output_base_dir = '/data/utsubo0/users/ruan/preprocessed_labels/'  # 输出文件夹基础目录

# 确保输出文件夹存在
os.makedirs(output_base_dir, exist_ok=True)

for organ in organs:
    # 定义路径
    early_tumor_dir = os.path.join(base_dir, f'{organ}_early_tumor')
    noearly_tumor_dir = os.path.join(base_dir, f'{organ}_noearly_tumor')
    output_dir = os.path.join(output_base_dir, f'{organ}_tumor')
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件列表，假设两个文件夹中的文件名完全相同
    file_list = os.listdir(early_tumor_dir)

    for filename in file_list:
        if filename.endswith('.nii.gz'):
            # 加载 early_tumor 和 noearly_tumor 的文件
            early_path = os.path.join(early_tumor_dir, filename)
            noearly_path = os.path.join(noearly_tumor_dir, filename)

            early_lbl = nib.load(early_path)
            noearly_lbl = nib.load(noearly_path)

            early_data = early_lbl.get_fdata()
            noearly_data = noearly_lbl.get_fdata()

            # 将 noearly_tumor 中的 2 改成 3
            if organ =='kidney':
                early_data[early_data == 2] = 2
                noearly_data[noearly_data == 2] = 3
            if organ =='liver':
                early_data[early_data == 2] = 4
                noearly_data[noearly_data == 2] = 5
            if organ =='pancreas':
                early_data[early_data == 2] = 6
                noearly_data[noearly_data == 2] = 7

            # 合并两个数组
            combined_data = np.maximum(early_data, noearly_data)

            print("uniquebefore:",np.unique(combined_data))
            combined_data = np.round(combined_data).astype(int)
            print("uniqueafter:",np.unique(combined_data))


            # 保存合并后的数据
            combined_img = nib.Nifti1Image(combined_data, early_lbl.affine, early_lbl.header)
            output_path = os.path.join(output_dir, filename)
            nib.save(combined_img, output_path)

    print(f"{organ} 的合并完成并保存到 {output_dir} 文件夹中。")
