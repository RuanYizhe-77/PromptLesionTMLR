# import os
# import random
#
# # Define custom directories for each dataset
# kits_ct_dir = '/data/utsubo0/users/ruan/05_KiTS/img/'          # 修改为实际的kits ct文件目录
# kits_label_dir = '/data/utsubo0/users/ruan/preprocessed_labels/kidney_tumor/'    # 修改为实际的kits label文件目录
# liver_ct_dir = '/data/utsubo0/users/ruan/Task03_Liver/imagesTr/'         # 修改为实际的liver ct文件目录
# liver_label_dir = '/data/utsubo0/users/ruan/preprocessed_labels/liver_tumor/'  # 修改为实际的liver label文件目录
# pancreas_ct_dir = '/data/utsubo0/users/ruan/Task07_Pancreas/imagesTr/'   # 修改为实际的pancreas ct文件目录
# pancreas_label_dir = '/data/utsubo0/users/ruan/preprocessed_labels/pancreas_tumor/' # 修改为实际的pancreas label文件目录
#
# # Create a list of directories for easy iteration
# datasets = [
#     (kits_ct_dir, kits_label_dir),
#     (liver_ct_dir, liver_label_dir),
#     (pancreas_ct_dir, pancreas_label_dir)
# ]
#
# # Initialize lists to store the paths
# tumor_paths = []
# label_paths = []
#
# # Traverse each dataset
# for ct_dir, label_dir in datasets:
#     for ct_file in os.listdir(ct_dir):
#         if not ct_file.startswith("._"):
#
#             if ct_file.startswith('img'):
#                 # Extract the index to match the label file
#                 index = ct_file.replace('img', '').replace('.nii.gz', '')
#                 ct_path = os.path.join(ct_dir, ct_file)
#                 label_path = os.path.join(label_dir, f'label{index}.nii.gz')
#
#                 # Ensure the corresponding label file exists
#                 if os.path.exists(label_path):
#                     tumor_paths.append(ct_path)
#                     label_paths.append(label_path)
#             else:
#                 # print("ct_file:",ct_file)
#                 organ = ct_file.split('_')[0]
#                 print(organ)
#                 index = ct_file.replace(organ, '').replace('.nii.gz', '')
#                 ct_path = os.path.join(ct_dir, ct_file)
#                 label_path = os.path.join(label_dir, f'{organ}{index}.nii.gz')
#                 print("ct_path:",ct_path)
#                 print("label_path:",label_path)
#
#                 # Ensure the corresponding label file exists
#                 if os.path.exists(label_path):
#                     tumor_paths.append(ct_path)
#                     label_paths.append(label_path)
#
# # Combine paths and shuffle
# data_pairs = list(zip(tumor_paths, label_paths))
# random.shuffle(data_pairs)
#
# # Split into train and validation sets with 8:2 ratio
# split_index = int(len(data_pairs) * 0.8)
# train_pairs = data_pairs[:split_index]
# val_pairs = data_pairs[split_index:]
#
# # Write train list
# list_dir = 'STEP2.DiffusionModel/cross_eval/alltumor_data_fold/'
# os.makedirs(list_dir, exist_ok=True)
# with open(os.path.join(list_dir,'real_tumor_train_1.txt'), 'w') as train_file:
#     for ct_path, label_path in train_pairs:
#         train_file.write(f"{ct_path} {label_path}\n")
#
# # Write validation list
# with open(os.path.join(list_dir,'real_tumor_val_1.txt'), 'w') as val_file:
#     for ct_path, label_path in val_pairs:
#         val_file.write(f"{ct_path} {label_path}\n")
#
# print("Train and validation lists saved as train_list.txt and val_list.txt")
import os
import random

# Define custom directories for each dataset
kits_ct_dir = '/data/utsubo0/users/ruan/05_KiTS/img/'          # kits ct文件目录
kits_label_dir = '/data/utsubo0/users/ruan/preprocessed_labels/kidney_tumor/'    # kits label文件目录
liver_ct_dir = '/data/utsubo0/users/ruan/Task03_Liver/imagesTr/'         # liver ct文件目录
liver_label_dir = '/data/utsubo0/users/ruan/preprocessed_labels/liver_tumor/'  # liver label文件目录
pancreas_ct_dir = '/data/utsubo0/users/ruan/Task07_Pancreas/imagesTr/'   # pancreas ct文件目录
pancreas_label_dir = '/data/utsubo0/users/ruan/preprocessed_labels/pancreas_tumor/' # pancreas label文件目录

# Dataset list for easy iteration
datasets = [
    (kits_ct_dir, kits_label_dir, 'kidney'),
    (liver_ct_dir, liver_label_dir, 'liver'),
    (pancreas_ct_dir, pancreas_label_dir, 'pancreas')
]

# Initialize dictionaries to store paths
tumor_paths = {dataset: [] for _, _, dataset in datasets}
label_paths = {dataset: [] for _, _, dataset in datasets}

# Traverse each dataset and collect paths
for ct_dir, label_dir, dataset_name in datasets:
    for ct_file in os.listdir(ct_dir):
        if not ct_file.startswith("._"):
            if ct_file.startswith('img'):
                index = ct_file.replace('img', '').replace('.nii.gz', '')
                ct_path = os.path.join(ct_dir, ct_file)
                label_path = os.path.join(label_dir, f'label{index}.nii.gz')

                if os.path.exists(label_path):
                    tumor_paths[dataset_name].append(ct_path)
                    label_paths[dataset_name].append(label_path)
            else:
                organ = ct_file.split('_')[0]
                index = ct_file.replace(organ, '').replace('.nii.gz', '')
                ct_path = os.path.join(ct_dir, ct_file)
                label_path = os.path.join(label_dir, f'{organ}{index}.nii.gz')

                if os.path.exists(label_path):
                    tumor_paths[dataset_name].append(ct_path)
                    label_paths[dataset_name].append(label_path)

# Step 1: Initial split of train and validation sets without repetition
train_pairs, val_pairs = [], []
for dataset_name in tumor_paths:
    data_pairs = list(zip(tumor_paths[dataset_name], label_paths[dataset_name]))
    random.shuffle(data_pairs)

    # Split into 80% train and 20% validation
    split_index = int(len(data_pairs) * 0.8)
    train_subset = data_pairs[:split_index]
    val_subset = data_pairs[split_index:]

    train_pairs.extend(train_subset)
    val_pairs.extend(val_subset)

# Step 2: Balance the train set by increasing each dataset's sample count to max_sample_count
# Find the maximum sample count among the datasets in the train set
max_sample_count = max(
    len([pair for pair in train_pairs if dataset_name in pair[1]])
    for _, _, dataset_name in datasets
)

# Create a balanced train set by increasing each dataset's sample count to max_sample_count
balanced_train_pairs = []
for dataset_name in tumor_paths:
    # Select pairs where the label path contains the dataset name
    dataset_train_pairs = [pair for pair in train_pairs if dataset_name in pair[1]]

    if len(dataset_train_pairs) < max_sample_count:
        # Randomly sample with replacement to reach max_sample_count
        repeats_needed = max_sample_count - len(dataset_train_pairs)
        dataset_train_pairs += random.choices(dataset_train_pairs, k=repeats_needed)

    balanced_train_pairs.extend(dataset_train_pairs)

# Shuffle again to mix across datasets
random.shuffle(balanced_train_pairs)

# Write train list
list_dir = 'STEP2.DiffusionModel/cross_eval/alltumor_data_fold/'
os.makedirs(list_dir, exist_ok=True)
with open(os.path.join(list_dir, 'real_tumor_train_1.txt'), 'w') as train_file:
    for ct_path, label_path in balanced_train_pairs:
        train_file.write(f"{ct_path} {label_path}\n")

# Write validation list (original validation split, without balancing)
with open(os.path.join(list_dir, 'real_tumor_val_1.txt'), 'w') as val_file:
    for ct_path, label_path in val_pairs:
        val_file.write(f"{ct_path} {label_path}\n")

print("Balanced train and validation lists saved as real_tumor_train_1.txt and real_tumor_val_1.txt")
