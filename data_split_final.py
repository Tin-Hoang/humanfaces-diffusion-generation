import os
import shutil
from PIL import Image
import numpy as np

# --------------------------
# CONFIGURATION
# --------------------------
image_dir = "./data/CelebAMask-HQ/CelebA-HQ-img"
mask_dir = "./data/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
mapping_file = "./data/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt"
partition_file = "./data/CelebAMask-HQ/list_eval_partition.txt"
attribute_file = "./data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
pose_file = "./data/CelebAMask-HQ/CelebAMask-HQ-pose-anno.txt"

output_base = "./data/CelebAMask-HQ-SPLIT"
split_txt_dir = os.path.join(output_base, "splits")
split_img_dir = os.path.join(output_base, "images_and_masks")
split_anno_dir = os.path.join(output_base, "annotations")
os.makedirs(split_txt_dir, exist_ok=True)
os.makedirs(split_img_dir, exist_ok=True)
os.makedirs(split_anno_dir, exist_ok=True)

# Define parts and labels
part_order = [
    "hair", "hat", "eyeglasses", "eyes", "eyebrows",
    "nose", "mouth", "lips", "teeth", "earrings"
]
label_mapping = {name: idx + 1 for idx, name in enumerate(part_order)}  # Avoid 0 for background

# --------------------------
# STEP 1: Generate HQ split IDs (2700 train + 300 val + 300 test)
# --------------------------
hq_to_celebA = {}
with open(mapping_file, "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            hq_id, _, celebA_file = parts
            celebA_file = celebA_file.zfill(9)
            hq_to_celebA[celebA_file] = int(hq_id)

celebA_partition = {}
with open(partition_file, "r") as f:
    for line in f:
        fname, split = line.strip().split()
        celebA_partition[fname] = int(split)

train_ids, val_ids, test_ids = [], [], []

for filename, hq_id in hq_to_celebA.items():
    split = celebA_partition.get(filename)
    if split == 0:
        train_ids.append(hq_id)
    elif split == 1:
        val_ids.append(hq_id)
    elif split == 2:
        test_ids.append(hq_id)

splits = {
    "train": sorted(train_ids)[:2700],
    "val": sorted(val_ids)[:300],
    "test": sorted(test_ids)[:300],
}

# --------------------------
# STEP 2: Write split ID files (zero-padded for consistency)
# --------------------------
for split_name in ["train", "val", "test"]:
    path = os.path.join(split_txt_dir, f"{split_name}_ids.txt")
    with open(path, "w") as f:
        for idx in range(len(splits[split_name])):
            f.write(f"{idx:05d}\n")

# --------------------------
# STEP 3: Combine part masks into a single semantic mask
# --------------------------
def combine_part_masks(mask_dir, prefix):
    combined = np.zeros((512, 512), dtype=np.uint8)
    for part in part_order:
        fname = f"{prefix}_{part}.png"
        fpath = os.path.join(mask_dir, fname)
        if os.path.exists(fpath):
            mask = np.array(Image.open(fpath).convert("L"))
            bin_mask = (mask > 0).astype(np.uint8)
            combined[bin_mask == 1] = label_mapping[part]
    return Image.fromarray(combined)

# --------------------------
# STEP 4: Copy images and create combined masks
# --------------------------
def copy_files(ids, split_name):
    img_out = os.path.join(split_img_dir, split_name, "images")
    mask_out = os.path.join(split_img_dir, split_name, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    for new_idx, original_idx in enumerate(ids):
        new_id_str = f"{new_idx:05d}"
        orig_img = os.path.join(image_dir, f"{original_idx}.jpg")
        new_img = os.path.join(img_out, f"{new_id_str}.jpg")

        if os.path.exists(orig_img):
            shutil.copyfile(orig_img, new_img)
        else:
            print(f"[Warning] Missing image: {orig_img}")

        mask_subdir = os.path.join(mask_dir, str(original_idx // 2000))
        prefix = f"{original_idx:05d}"
        if os.path.exists(mask_subdir):
            combined = combine_part_masks(mask_subdir, prefix)
            combined.save(os.path.join(mask_out, f"{new_id_str}_mask.png"))
        else:
            print(f"[Warning] Missing mask folder: {mask_subdir}")

    print(f"✅ Copied {len(ids)} images and combined masks to '{split_name}/'")

copy_files(splits["train"], "train")
copy_files(splits["val"], "val")
copy_files(splits["test"], "test")

# --------------------------
# STEP 5: Fix and split annotation files using zero-padded image names
# --------------------------
def split_annotation(file_path, splits, prefix):
    with open(file_path, "r") as f:
        f.readline()  # Skip original number of images
        header_line = f.readline().strip()
        lines = f.readlines()

    for split in ["train", "val", "test"]:
        ids = splits[split]
        output_lines = []
        for new_idx, original_idx in enumerate(ids):
            if original_idx < len(lines):
                parts = lines[original_idx].strip().split()
                new_name = f"{new_idx:05d}.jpg"
                attribute_values = " ".join(parts[1:])
                output_lines.append(f"{new_name} {attribute_values}\n")
            else:
                print(f"[Warning] Annotation index {original_idx} is out of bounds.")

        out_path = os.path.join(split_anno_dir, f"{prefix}_{split}.txt")
        with open(out_path, "w") as f_out:
            f_out.write(f"{len(output_lines)}\n")
            f_out.write(header_line + "\n")
            f_out.writelines(output_lines)

        print(f"✅ Wrote {len(output_lines)} lines to {prefix}_{split}.txt")

split_annotation(attribute_file, splits, "attribute")
split_annotation(pose_file, splits, "pose")

print("\n DONE! Created dataset with 2700 train, 300 val, 300 test and combined semantic masks.")
