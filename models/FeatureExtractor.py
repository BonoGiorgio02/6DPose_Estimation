from torch import nn
from torchvision import models

class FeatureExtractor(nn.Module):
    """Extract features from images using ResNet specified backbone
        returns a tensor of shape (batch_size, feature_dim)
    """

    def __init__(self, backbone='resnet18', weights= None):
        """Import Resnet architecture pretrained

        Args:
            backbone (str, optional): Name of the Resnet model. Defaults to 'resnet18'.
            weights (str, optional): Pretrained weights of the Resnet
        """
        super(FeatureExtractor, self).__init__()
        self.weights= weights

        if backbone == 'resnet18':
            # load pretrained weights if self.weights=None
            self.backbone = models.resnet18(weights= "ResNet18_Weights.IMAGENET1K_V1" if self.weights==None else self.weights)
            self.feature_dim = 512 # feature size of fully connected layer input

        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights= "ResNet50_Weights.IMAGENET1K_V2"  if self.weights==None else self.weights)
            self.feature_dim = 2048

        # select all layers except the last one (fully connected layers)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # freeze layers
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1) # reshape tensor to be (batch_size, feature_dim), features.size(0) is batch_size
        return features

    def get_feature_dim(self):
        return self.feature_dim