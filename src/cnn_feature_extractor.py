import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor using a pretrained ResNet18 backbone.
    Input: RGB image (3x224x224)
    Output: 512-dim feature vector per frame
    """
    def __init__(self, pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Remove the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # output: (batch, 512, 1, 1)

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        features = self.feature_extractor(x)  # (batch, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 512)
        return features

# Example usage:
if __name__ == "__main__":
    model = CNNFeatureExtractor(pretrained=True)
    dummy = torch.randn(4, 3, 224, 224)  # batch of 4 images
    feats = model(dummy)
    print(feats.shape)  # should be (4, 512)
