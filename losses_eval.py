import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian_Neg_Pearson_Loss(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super(Gaussian_Neg_Pearson_Loss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self.create_gaussian_kernel_3d(
            kernel_size, sigma)

    def create_gaussian_kernel_3d(self, kernel_size, sigma):
        """Creates a 3D Gaussian kernel for smoothing."""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        # Shape for 3D conv
        return kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    def apply_gaussian_smoothing(self, X):
        """Applies 3D Gaussian smoothing to the input volume."""
        B, C, D, H, W = X.shape  # (batch, channels, depth, height, width)
        # Move kernel to correct device
        kernel = self.gaussian_kernel.to(X.device)
        smoothed_X = F.conv3d(
            X, kernel, padding=self.kernel_size // 2, groups=C)
        return smoothed_X

    def forward(self, X, Y):
        # Ensure inputs have shape (batch, 1, depth, height, width)
        if X.dim() == 4:  # If missing channel dimension, add one
            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)

        # Apply 3D Gaussian smoothing
        X = self.apply_gaussian_smoothing(X)
        Y = self.apply_gaussian_smoothing(Y)

        # Flatten to vector format (batch_size, num_elements)
        X = X.view(X.shape[0], -1)
        Y = Y.view(Y.shape[0], -1)

        # Check for NaNs
        assert not torch.any(torch.isnan(X)), "X contains NaNs"
        assert not torch.any(torch.isnan(Y)), "Y contains NaNs"

        # Compute means
        mean_X = X.mean(dim=1, keepdim=True)
        mean_Y = Y.mean(dim=1, keepdim=True)

        # Compute standard deviations
        std_X = X.std(dim=1, keepdim=True) + 1e-5  # Avoid division by zero
        std_Y = Y.std(dim=1, keepdim=True) + 1e-5

        # Normalize
        X_norm = (X - mean_X) / std_X
        Y_norm = (Y - mean_Y) / std_Y

        # Compute Pearson correlation
        correlation = torch.sum(X_norm * Y_norm, dim=1) / (X.size(1) - 1)

        # Compute loss: 1 - mean correlation
        loss = 1 - correlation.mean()
        return loss


class Neg_Pearson_Loss(nn.Module):
    def __init__(self):
        super(Neg_Pearson_Loss, self).__init__()

    def forward(self, X, Y):
        # Check for NaNs
        assert not torch.any(torch.isnan(X)), "X contains NaNs"
        assert not torch.any(torch.isnan(Y)), "Y contains NaNs"

        # Normalize: Subtract mean
        X = X - X.mean(dim=1, keepdim=True)
        Y = Y - Y.mean(dim=1, keepdim=True)

        # Standardize: Divide by standard deviation
        X = X / (X.std(dim=1, keepdim=True) + 1e-5)
        Y = Y / (Y.std(dim=1, keepdim=True) + 1e-5)

        # Compute Pearson correlation
        correlation = torch.sum(X * Y, dim=1) / X.size(1)

        # Compute loss: 1 - mean correlation
        loss = 1 - correlation.mean()
        return loss
