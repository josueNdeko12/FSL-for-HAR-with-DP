import numpy as np
import torch

# Function to add Gaussian noise for differential privacy
def add_gaussian_noise(data, epsilon, delta):
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = torch.normal(0, sigma, data.size()).to(data.device)
    return data + noise

# Function to clip gradients
def clip_gradients(data, clip_norm):
    norm = torch.norm(data, dim=1, keepdim=True)
    norm = torch.clamp(norm, max=clip_norm)
    return data / norm * clip_norm