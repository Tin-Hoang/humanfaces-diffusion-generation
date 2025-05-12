import os
from typing import List, Tuple, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from diffusion_models.datasets.data_utils import get_inference_transform


class AttributeDataset(Dataset):
    """Dataset class for loading images with attribute labels.

    This dataset loads images from a directory and their corresponding attribute labels
    from a CSV-formatted text file. The attribute labels are binary (-1 for no, 1 for yes).

    Args:
        image_dir (str): Directory containing the image files
        attribute_label_path (str): Path to the attribute label file
        image_size (int): Size to resize images to (both height and width)
        transform (Optional[transforms.Compose]): Optional transforms to apply to images
        mask_dir (Optional[str]): Directory containing the segmentation masks
     """


    def __init__(
        self,
        image_dir: str,
        attribute_label_path: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        mask_dir: Optional[str] = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform or get_inference_transform(image_size)

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]) if mask_dir else None

        # Get list of existing images in the directory
        existing_images = set(f"{f}" for f in os.listdir(image_dir) if f.endswith('.jpg'))
        print(f"Found {len(existing_images)} images in directory")

        # Read the attribute file
        # Skip the first line (number of images) and use the second line as headers
        print(f"Reading attribute file: {attribute_label_path}")

        # First read the headers
        with open(attribute_label_path, 'r') as f:
            # Skip first line (number of images)
            f.readline()
            # Get header line
            header_line = f.readline().strip()
            # Get the attribute names
            attribute_names = header_line.split()

        # Now read the data, skipping the first two lines
        self.attributes_df = pd.read_csv(
            attribute_label_path,
            skiprows=2,
            sep=r'\s+',
            header=None,
            names=['image_id'] + attribute_names,
            dtype=str
        )
        self.attributes_df['image_id'] = self.attributes_df['image_id'].apply(
            lambda x: f"{x}.jpg" if not x.endswith('.jpg') else x
        )
        for col in self.attributes_df.columns[1:]:
            self.attributes_df[col] = pd.to_numeric(self.attributes_df[col], errors='coerce')
            self.attributes_df[col] = self.attributes_df[col].map({-1: 0, 1: 1})

        self.attributes_df = self.attributes_df[self.attributes_df['image_id'].isin(existing_images)]

        if len(self.attributes_df) == 0:
            print("Debugging information:")
            print(f"Total images in attribute file: {len(self.attributes_df)}")
            print(f"Sample of image_ids in attribute file before filtering:")
            print(self.attributes_df['image_id'].head() if len(self.attributes_df) > 0 else "No images found")
            print("Sample of image_ids in directory:")
            print(list(existing_images)[:5])

            sample_attr_ids = set(self.attributes_df['image_id'].head().tolist())
            sample_dir_ids = set(list(existing_images)[:5])
            print("Checking for exact matches:")
            print(f"Attribute file IDs: {sample_attr_ids}")
            print(f"Directory IDs: {sample_dir_ids}")
            print(f"Common IDs: {sample_attr_ids.intersection(sample_dir_ids)}")

            raise ValueError(
                f"No matching images found between {image_dir} and {attribute_label_path}\n"
                f"Found {len(existing_images)} images in directory\n"
                f"First few images in directory: {list(existing_images)[:5]}\n"
                f"First few images in attribute file: {list(self.attributes_df['image_id'])[:5] if len(self.attributes_df) > 0 else 'No images found'}"
            )

        self.attribute_names = self.attributes_df.columns[1:].tolist()

        print(f"Final dataset size: {len(self.attributes_df)} images with attributes out of {len(existing_images)} images in directory")

    def __len__(self) -> int:
        return len(self.attributes_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.attributes_df.iloc[idx]
        image_id = row['image_id']
        image_path = os.path.join(self.image_dir, image_id)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        attributes = torch.from_numpy(row[1:].values.astype(np.float32))

        if self.mask_dir:
            combined_mask = torch.zeros((256, 256), dtype=torch.uint8)

            part_labels = [
                'hair', 'hat', 'eyeglasses', 'eyes', 'eyebrows',
                'nose', 'mouth', 'lips', 'teeth', 'earrings'
            ]

            base_id = image_id.replace('.jpg', '')
            for class_idx, part in enumerate(part_labels, start=1):
                part_filename = f"{base_id}_{part}.png"
                part_path = os.path.join(self.mask_dir, part_filename)

                if os.path.exists(part_path):
                    part_mask = Image.open(part_path).convert('RGB')
                    part_mask = self.mask_transform(part_mask)  # (3, H, W)
                    part_mask_gray = part_mask.mean(dim=0)  # (H, W)
                    combined_mask[part_mask_gray > 0.05] = class_idx  # 0.05 threshold to ignore near-black

            combined_mask = combined_mask.unsqueeze(0).float()
            return image, attributes, combined_mask
        else:
            return image, attributes

    def get_attribute_names(self) -> List[str]:
        return self.attribute_names

# DEBUGGING
if __name__ == "__main__":
    import sys
    from pathlib import Path

    try:
        dataset = AttributeDataset(
            image_dir="data/CelebA-HQ-split/test_300",
            attribute_label_path="data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt",
            mask_dir="data/CelebA-HQ-split/test_300_masks"
        )

        assert len(dataset) == 300, f"Expected length 300, got {len(dataset)}"
        attribute_names = dataset.get_attribute_names()
        expected_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
        assert attribute_names == expected_names, f"Expected {expected_names}, got {attribute_names}"

        image, attributes, mask = dataset[0]
        assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
        assert isinstance(attributes, torch.Tensor), "Attributes should be a torch.Tensor"
        assert isinstance(mask, torch.Tensor), "Mask should be a torch.Tensor"
        assert image.shape == (3, 256, 256), f"Expected image shape (3, 256, 256), got {image.shape}"
        assert attributes.shape == (40,), f"Expected attributes shape (40,), got {attributes.shape}"
        assert mask.shape == (1, 256, 256), f"Expected mask shape (1, 256, 256), got {mask.shape}"
        print(f"First mask unique class indices: {torch.unique(mask)}")
        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        sys.exit(1)
