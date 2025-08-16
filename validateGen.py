import random

# 指定输入和输出文件的路径
file1_path = '/home/mil/ruan/FreeLesion/STEP2.DiffusionModel/cross_eval/coronary_data_fold/real_lesion_train_0.txt'  # 第一个TXT文件路径
file2_path = '/home/mil/ruan/FreeLesion/STEP3.SegmentationModel/cross_eval/coronary_aug_data_fold/total_lesion.txt'  # 第二个TXT文件路径
output_path = '/home/mil/ruan/FreeLesion/STEP3.SegmentationModel/cross_eval/coronary_aug_data_fold/real_lesion_train_0.txt'  # 输出结果保存的TXT文件路径

# 读取第一个文件中的文件夹名
with open(file1_path, 'r') as file1:
    list1 = {line.strip() for line in file1}  # 使用集合以去重

# 读取第二个文件中的文件夹名
with open(file2_path, 'r') as file2:
    list2 = {line.strip() for line in file2}  # 使用集合以去重

# 找到两个列表中不重复的部分（对称差集）
unique_folders = list(list1.symmetric_difference(list2))

# 打乱不重复部分的顺序
random.shuffle(unique_folders)

# 筛选其中50%的文件夹名
half_count = len(unique_folders) // 2
selected_folders = unique_folders[:half_count]

# 将结果保存到新的TXT文件中
with open(output_path, 'w') as output_file:
    for folder in selected_folders:
        output_file.write(f"{folder}\n")

print(f"筛选的文件夹名已保存到 {output_path} 中。")
