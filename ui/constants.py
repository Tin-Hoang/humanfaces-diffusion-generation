"""Constants for the UI."""

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = "checkpoints/ddpm-celebahq-128-27000train-20250316_141247"

# Default checkpoint for attribute generation
DEFAULT_ATTRIBUTE_CHECKPOINT_DIR = "checkpoints/attribute_latentconditionalunet2d_256_ddim_27000train_20250325_235926"

# Default generation parameters
DEFAULT_NUM_STEPS = 100
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_IMAGES = 4

# Image dimensions
NOISE_IMAGE_SIZE = 128
GALLERY_COLUMNS = 2
GALLERY_ROWS = 2
GALLERY_HEIGHT = 400

ATTRIBUTES = {
    0: "5_o_Clock_Shadow",
    1: "Arched_Eyebrows",
    2: "Attractive",
    3: "Bags_Under_Eyes",
    4: "Bald",
    5: "Bangs",
    6: "Big_Lips",
    7: "Big_Nose",
    8: "Black_Hair",
    9: "Blond_Hair",
    10: "Blurry",
    11: "Brown_Hair",
    12: "Bushy_Eyebrows",
    13: "Chubby",
    14: "Double_Chin",
    15: "Eyeglasses",
    16: "Goatee",
    17: "Gray_Hair",
    18: "Heavy_Makeup",
    19: "High_Cheekbones",
    20: "Male",
    21: "Mouth_Slightly_Open",
    22: "Mustache",
    23: "Narrow_Eyes",
    24: "No_Beard",
    25: "Oval_Face",
    26: "Pale_Skin",
    27: "Pointy_Nose",
    28: "Receding_Hairline",
    29: "Rosy_Cheeks",
    30: "Sideburns",
    31: "Smiling",
    32: "Straight_Hair",
    33: "Wavy_Hair",
    34: "Wearing_Earrings",
    35: "Wearing_Hat",
    36: "Wearing_Lipstick",
    37: "Wearing_Necklace",
    38: "Wearing_Necktie",
    39: "Young",
}

# Get attribute names from ATTRIBUTES
ATTRIBUTE_NAMES = [ATTRIBUTES[attr] for attr in ATTRIBUTES]
