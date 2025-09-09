import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
from transformer_block import HemiretinalCrossAttentionBlock, HemiretinalNeuronAxonCrossAttentionBlock, NeuronAxonCrossAttentionBlock, EPABlock
import monai


def make_block(in_channels, out_channels, kernel_size, stride, padding, num_features):
    # make a 3D convolution
    block = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm3d(num_features=num_features), nn.ReLU())
    return block


def make_conv_block(in_channels, out_channels, kernel_size, stride, padding):
    # make a 3D convolution
    block = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm3d(num_features=out_channels), nn.ReLU())
    return block


def make_block_with_pool(in_channels, out_channels, kernel_size, stride, padding, num_features):
    # make a 3D convolution
    block = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm3d(num_features=num_features), nn.ReLU(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
    return block


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


class AI_C_NET_3D(nn.Module):
    def __init__(self, input_size, model_params):
        super(AI_C_NET_3D, self).__init__()
        att_type = model_params['att_type']
        att_ind = model_params['att_ind']
        max_pool_ind = model_params['max_pool_ind']

        self.layers = nn.ModuleList()

        self.gradcam_layer_num = model_params['gradcam_layer_num'] - 1
        self.CARE_layer_num = model_params['CARE_layer_num']
        self.CARE_layer = None

        def forward_hook(module, input, output):
            self.hooked_activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        for i in range(0, model_params['num_conv_layers']):
            self.layers.append(make_conv_block(in_channels=model_params['conv_in_channels'][i], out_channels=model_params['conv_out_channels'][i],
                               kernel_size=model_params['conv_kernel_size'][i], stride=model_params['conv_stride'][i], padding=model_params['conv_padding'][i]))
            if self.gradcam_layer_num == i:
                # Convolution within sequential block
                conv3d_layer = self.layers[-1][0]
                conv3d_layer.register_forward_hook(forward_hook)
                conv3d_layer.register_backward_hook(backward_hook)

            if i+1 in att_ind:

                if att_type == 'Hemiretinal':
                    self.layers.append(nn.Sequential(HemiretinalCrossAttentionBlock(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                                    dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
                elif att_type == 'HemiretinalNeuronAxon':
                    self.layers.append(nn.Sequential(HemiretinalNeuronAxonCrossAttentionBlock(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                                              dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
                elif att_type == 'NeuronAxon':
                    self.layers.append(nn.Sequential(NeuronAxonCrossAttentionBlock(hidden_size=model_params['conv_out_channels'][i], num_heads=4,
                                                                                   dropout_rate=0.15), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
                elif att_type == 'EPA':
                    ind = next((ind for ind, val in enumerate(
                        att_ind) if i + 1 == val), len(att_ind))
                    factor = 2 ** (ind + 1)
                    att_input_size = (
                        input_size[0] // factor) * (input_size[1] // factor) * (input_size[2] // factor)
                    self.layers.append(nn.Sequential(EPABlock(input_size=att_input_size, hidden_size=32, proj_size=64,
                                       num_heads=4, dropout_rate=0.15, pos_embed=False), torch.nn.MaxPool3d(kernel_size=(2, 2, 2))))
            # Store reference to CARE layer for attention visualization
            if i+1 == self.CARE_layer_num:
                self.CARE_layer = self.layers[-1]
            if i+1 in max_pool_ind:

                self.layers.append(torch.nn.MaxPool3d(kernel_size=(2, 2, 2)))

        conv_down = np.sum(
            model_params['conv_stride'])-len(model_params['conv_stride'])
        stride = np.array(input_size) / np.power(2,
                                                 len(att_ind)+len(max_pool_ind)+conv_down)
        stride = stride.astype(np.int64)

        kernel_size = int(np.min(stride))
        self.GAP = nn.AvgPool3d(kernel_size=kernel_size, stride=tuple(stride))
        self.dense = nn.Linear(
            in_features=model_params['conv_out_channels'][-1], out_features=2)

        self.Softmax = torch.nn.Softmax(dim=1)
        # Grad-CAM variables for convolutional heatmaps
        # These capture activations and gradients from the specified conv layer for Grad-CAM visualization
        self.hooked_activations = None
        self.gradients = None

    def forward(self, x):
        batch_size = x.shape[0]
        # CARE (Cross-Attention Representation) - captures attention from transformer blocks
        attn_return = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Store attention output from the specified CARE layer for visualization
            if layer == self.CARE_layer:
                attn_return = x
        x = torch.squeeze(self.GAP(x))
        x = self.dense(x)
        # make sure 2D tensor
        x = x.view(batch_size, -1)
        x = self.Softmax(x)

        if self.CARE_layer is not None:
            # Process CARE attention representation for visualization
            # Sum across channels to generate attention heatmap
            attn_return = torch.mean(attn_return, dim=1)

            # Apply ReLU and normalization to CARE attention maps
            attn_return = torch.nn.functional.relu(attn_return)
            max_vals = torch.amax(attn_return, dim=(1, 2, 3), keepdim=True)
            attn_return = attn_return / (max_vals + 1e-5)

        # Returns: (predictions, CARE_attention_maps)
        # CARE attention maps show which spatial regions the transformer blocks focus on
        return x, attn_return

    def compute_gradcam_batch(self):
        """
        Compute Grad-CAM heatmaps for the entire batch.
        These are CONVOLUTIONAL heatmaps that show which spatial regions 
        the CNN layers focus on for classification decisions.

        Returns:
            torch.Tensor: Grad-CAM heatmaps for each input in the batch.
            These differ from CARE attention maps - Grad-CAM shows CNN focus,
            while CARE shows transformer attention focus.
        """
        if self.hooked_activations is None or self.gradients is None:
            raise ValueError(
                "Forward and backward hooks have not captured data.")

        # Grad-CAM computation: weight activations by gradients
        # Global average pooling over gradients to get importance weights
        # Shape: (batch_size, num_channels)
        pooled_grads = torch.mean(self.gradients, dim=(2, 3, 4))

        # Expand dimensions to match activation shape for broadcasting
        pooled_grads = pooled_grads[:, :, None, None, None]

        # Weight activations by pooled gradients (Grad-CAM core computation)
        weighted_activations = self.hooked_activations * pooled_grads

        # Sum across channels to generate Grad-CAM heatmap
        # This creates spatial attention maps showing CNN layer focus
        heatmaps = torch.sum(weighted_activations, dim=1)

        # Apply ReLU and normalization for visualization
        heatmaps = torch.nn.functional.relu(heatmaps)
        max_vals = torch.amax(heatmaps, dim=(1, 2, 3), keepdim=True)
        heatmaps = heatmaps / (max_vals + 1e-5)

        # Return Grad-CAM heatmaps (CNN-based attention) vs CARE attention (transformer-based)
        return heatmaps
