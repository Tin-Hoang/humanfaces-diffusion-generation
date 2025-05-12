import os
import numpy as np
from PIL import Image
from scipy import linalg
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader, Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, fname) 
                      for fname in os.listdir(folder) 
                      if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

class InceptionV3Feature(torch.nn.Module):
    """Inception v3 model for FID computation (using pool3 features)"""
    def __init__(self):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        # Extract up to pool3 layer (final avg pooling layer before classifier)
        self.block1 = torch.nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, 
            inception.Conv2d_2b_3x3, torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = torch.nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block3 = torch.nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e
        )
        self.block4 = torch.nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        # Disable gradient computation for efficiency
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.squeeze(-1).squeeze(-1)  # Return flattened 2048-dim feature vector

def get_activations(folder, model, batch_size=50, dims=2048, device='cpu', num_workers=4):
    """Compute Inception activations for all images in a folder."""
    # Standard preprocessing for Inception v3
    transform = transforms.Compose([
        transforms.Resize(299),  # Resize the smaller edge to 299
        transforms.CenterCrop(299),  # Center crop to 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageFolderDataset(folder, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers,
                            drop_last=False)
    
    model.eval()
    
    # Pre-allocate output array
    features = np.empty((len(dataset), dims))
    
    # Process batches
    idx = 0
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            batch_features = model(batch)
        
        # Move to CPU and convert to numpy
        batch_features = batch_features.cpu().numpy()
        batch_size = batch_features.shape[0]
        features[idx:idx + batch_size] = batch_features
        idx += batch_size
    
    return features

def calculate_statistics(act):
    """Calculate mean and covariance statistics from activations."""
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate the Frechet Distance between multivariate Gaussians."""
    diff = mu1 - mu2
    
    # Calculate sqrt(A*B) - numerically stable version
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Fix numerical issues
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produces singular matrix"
        print(f"WARNING: {msg}")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    # Calculate FID formula: ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    return (diff.dot(diff) + 
            np.trace(sigma1) + 
            np.trace(sigma2) - 
            2 * tr_covmean)

def calculate_fid(real_folder, gen_folder, batch_size=50, device=None):
    """Calculate FID between images in two folders."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create Inception model specifically for FID
    model = InceptionV3Feature().to(device)
    
    # Get features
    print(f"Calculating features for real images from {real_folder}")
    real_features = get_activations(real_folder, model, batch_size=batch_size, device=device)
    
    print(f"Calculating features for generated images from {gen_folder}")
    gen_features = get_activations(gen_folder, model, batch_size=batch_size, device=device)
    
    # Check if we have enough images
    if len(real_features) < 2 or len(gen_features) < 2:
        raise ValueError("Need at least 2 images per folder to calculate FID")
    
    print(f"Calculating statistics...")
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_gen, sigma_gen = calculate_statistics(gen_features)
    
    print(f"Calculating FID...")
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid_value

if __name__ == '__main__':
    real_dir = '/test_image_path'
    gen_dir = '/generated_image_path'
    
    # Calculate FID
    fid_score = calculate_fid(real_dir, gen_dir, batch_size=50)
    print(f'FID score: {fid_score:.4f}')