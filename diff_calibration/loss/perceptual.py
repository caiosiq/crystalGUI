
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=None, use_gram=True, resize=False, weights=None):
        """
        VGG-19 based Perceptual Loss (Feature Matching & Style Loss).
        
        Args:
            feature_layers (list): Indices/Names of layers to extract features from.
                                   Default: ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
            use_gram (bool): If True, computes Gram Matrix loss (Style). 
                             If False, computes Feature Matching loss (Content).
            resize (bool): If True, resizes input to 224x224 (VGG standard). 
                           Set False if input is already large enough to save VRAM/compute.
            weights (list): Weights for each layer's loss contribution.
                            Default: [1.0, 1.0, 1.0, 1.0, 1.0]
                            For microscopy style, recommend [1.0, 0.8, 0.5, 0.2, 0.1] to emphasize texture.
        """
        super().__init__()
        
        # Load VGG19 pretrained on ImageNet
        # .features gives us the convolutional part
        vgg_pretrained = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        # Mapping for VGG19 (features):
        # relu1_1 (idx 1)
        # relu2_1 (idx 6)
        # relu3_1 (idx 11)
        # relu4_1 (idx 20)
        # relu5_1 (idx 29)
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained[x])
            
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.use_gram = use_gram
        self.resize = resize
        
        # Layer weights
        if weights is None:
            # Emphasize early layers for texture/style
            self.weights = [1.0, 0.8, 0.4, 0.2, 0.1]
        else:
            self.weights = weights
        
        # Normalization for VGG (ImageNet mean/std)
        # Expecting input in range [0, 1]
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # Input/Target should be (N, 3, H, W)
        # STRICTLY EXPECTS RANGE [0, 1]
        
        # If input is single channel (grayscale), replicate to 3
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
            
        # Normalize with ImageNet stats
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        loss = 0.0
        
        # Forward Pass through slices
        # Slice 1 (relu1_1)
        h_x = self.slice1(input)
        h_y = self.slice1(target)
        loss += self.compute_layer_loss(h_x, h_y) * self.weights[0]
        
        # Slice 2 (relu2_1)
        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        loss += self.compute_layer_loss(h_x, h_y) * self.weights[1]
        
        # Slice 3 (relu3_1)
        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        loss += self.compute_layer_loss(h_x, h_y) * self.weights[2]

        # Slice 4 (relu4_1)
        h_x = self.slice4(h_x)
        h_y = self.slice4(h_y)
        loss += self.compute_layer_loss(h_x, h_y) * self.weights[3]
        
        # Slice 5 (relu5_1) - Optional, sometimes too deep for texture
        # h_x = self.slice5(h_x)
        # h_y = self.slice5(h_y)
        # loss += self.compute_layer_loss(h_x, h_y) * self.weights[4]
        
        return loss

    def compute_layer_loss(self, x, y):
        if self.use_gram:
            return self.gram_loss(x, y)
        else:
            return F.mse_loss(x, y)

    def gram_matrix(self, x):
        N, C, H, W = x.size()
        features = x.view(N, C, H * W)
        # G = F * F^T
        G = torch.bmm(features, features.transpose(1, 2))
        # Normalize by number of elements to avoid exploding gradients
        return G / (C * H * W)

    def gram_loss(self, x, y):
        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        return F.mse_loss(Gx, Gy)
