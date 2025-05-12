from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import numpy as np

class CelebAMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.filenames[idx].replace('.jpg', '_mask.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)  # Apply transforms to image only

        mask = mask.resize((512, 512), resample=Image.NEAREST)  # Preserve class labels
        mask = torch.from_numpy(np.array(mask)).long()  # Convert to LongTensor of shape [H, W]

        return {"pixel_values": image, "labels": mask}


from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=11,        # 10 classes + background
    ignore_mismatched_sizes=True
)

from transformers import TrainingArguments

from torchvision import transforms

common_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

train_dataset = CelebAMaskDataset(
    image_dir="data/CelebAMask-HQ-SPLIT/images_and_masks/train/images",
    mask_dir="data/CelebAMask-HQ-SPLIT/images_and_masks/train/masks",
    transform=common_transforms
)

val_dataset = CelebAMaskDataset(
    image_dir="data/CelebAMask-HQ-SPLIT/images_and_masks/val/images",
    mask_dir="data/CelebAMask-HQ-SPLIT/images_and_masks/val/masks",
    transform=common_transforms
)

args = TrainingArguments(
    output_dir="./segformer-celebamask",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=30,
    learning_rate=5e-5,
    save_steps=1000,
    eval_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
)
trainer.train()

trainer.save_model("./segformer-celebamask-finetuned")



