def split_dataset_by_organ(input_file, output_dir_kidney, output_dir_liver, output_dir_pancreas, txt_type="train"):
    import os

    # 创建输出目录
    os.makedirs(output_dir_kidney, exist_ok=True)
    os.makedirs(output_dir_liver, exist_ok=True)
    os.makedirs(output_dir_pancreas, exist_ok=True)


    # 器官名
    organs = ["kidney", "liver", "pancreas"]

    # 初始化器官分组
    organ_groups = {organ: [] for organ in organs}

    # 读取文件内容
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 遍历每一行，根据器官名分类
    for line in lines:
        ct_path, label_path = line.strip().split()
        for organ in organs:
            if organ in label_path:
                organ_groups[organ].append(line)
                break  # 每行只会归属到一个器官

    # 将分组后的数据写入新文件
    for organ, organ_lines in organ_groups.items():
        if organ == "kidney":
            output_file = os.path.join(output_dir_kidney,f"real_tumor_{txt_type}_0.txt")
            print(f"结果保存在 {output_file} 中。")
        elif organ == "liver":
            output_file = os.path.join(output_dir_liver,f"real_tumor_{txt_type}_0.txt")
            print(f"结果保存在 {output_file} 中。")
        elif organ == "pancreas":
            output_file = os.path.join(output_dir_pancreas,f"real_tumor_{txt_type}_0.txt")
            print(f"结果保存在 {output_file} 中。")
        with open(output_file, 'w') as file:
            file.writelines(organ_lines)



# 示例调用
input_txt_train = "/home/mil/ruan/FreeLesion/STEP2.DiffusionModel/cross_eval/alltumor_data_fold/real_tumor_train_0.txt"
input_txt_val = "/home/mil/ruan/FreeLesion/STEP2.DiffusionModel/cross_eval/alltumor_data_fold/real_tumor_val_0.txt"
# 输入的txt文件路径
output_dir_kidney = "/home/mil/ruan/FreeLesion/STEP3.SegmentationModel/cross_eval/kidney_aug_data_fold/"
output_dir_liver = "/home/mil/ruan/FreeLesion/STEP3.SegmentationModel/cross_eval/liver_aug_data_fold/"
output_dir_pancreas = "/home/mil/ruan/FreeLesion/STEP3.SegmentationModel/cross_eval/pancreas_aug_data_fold/"
# 输出目录
split_dataset_by_organ(input_txt_train, output_dir_kidney, output_dir_liver, output_dir_pancreas, txt_type="train")
split_dataset_by_organ(input_txt_val, output_dir_kidney, output_dir_liver, output_dir_pancreas, txt_type="val")
