from torch import nn
from models.FeatureExtractor import FeatureExtractor

class PosePredictorModelAlternative(nn.Module):
  """After Resnet, add a second NN to predict translation and rotation
        returns translation and rotation predictions
  """

  def __init__(self, backbone='resnet18', hidden_dim=512):
        """Initialize model and import Resnet backbone

        Args:
            backbone (str, optional): Name of the Resnet architecture to import. Defaults to 'resnet18'. Otherwise resnet50
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 512.
        """
        super(PosePredictorModelAlternative, self).__init__()

        # Feature extractor, import resnet architecture
        self.feature_extractor = FeatureExtractor(backbone)
        feature_dim = self.feature_extractor.get_feature_dim()

        # Output heads separate for translation and rotation
        self.translation_head = nn.Linear(feature_dim, 3)
        self.rotation_head = nn.Linear(feature_dim, 4)

  def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        translation = self.translation_head(features)
        rotation_flat = self.rotation_head(features)

        batch_size = rotation_flat.size(0)
        rotation = rotation_flat.view(batch_size, 4)

        return translation, rotation