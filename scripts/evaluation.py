"""Script to calculate FID score between two directories of images."""

import argparse
import torch
from diffusion_models.utils.metrics import calculate_fid_from_folders


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID score between two directories of images")
    parser.add_argument("--real-dir", type=str, required=True,
                      help="Directory containing real images")
    parser.add_argument("--fake-dir", type=str, required=True,
                      help="Directory containing generated/fake images")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for loading images to fid.update")
    parser.add_argument("--image-size", type=int, default=None,
                      help="Size to resize images to (optional)")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (e.g. 'cuda:0', 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate FID
    fid_score = calculate_fid_from_folders(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size
    )


if __name__ == "__main__":
    main() 