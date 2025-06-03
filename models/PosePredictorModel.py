from torch import nn
from models.FeatureExtractor import FeatureExtractor

class PosePredictorModel(nn.Module):
    """After Resnet, add a second NN to predict translation and rotation
        returns translation and rotation predictions
    """

    def __init__(self, backbone='resnet18', hidden_dim=512):
        """Initialize model and import Resnet backbone

        Args:
            backbone (str, optional): Name of the Resnet architecture to import. Defaults to 'resnet18'. Otherwise resnet50
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 512.
        """
        super(PosePredictorModel, self).__init__()

        # Feature extractor, import resnet architecture
        self.feature_extractor = FeatureExtractor(backbone)
        feature_dim = self.feature_extractor.get_feature_dim()

        # Fully connected layers for the pose prediction
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(feature_dim, hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_dim//2, hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_dim//2, hidden_dim//4),
        #     nn.ReLU()
        # )

        self.fc_layers = nn.Sequential(
          nn.Linear(feature_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
          nn.Dropout(0.3),

          nn.Linear(hidden_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
          nn.Dropout(0.3),

          nn.Linear(hidden_dim, hidden_dim // 2), # half the dimension
          nn.BatchNorm1d(hidden_dim // 2), # the number of features is reduced from previous Linear
          nn.ReLU(),
          nn.Dropout(0.3),

          nn.Linear(hidden_dim // 2, hidden_dim // 4),
          nn.BatchNorm1d(hidden_dim // 4),
          nn.ReLU()
        )

        # Output heads separate for translation and rotation
        self.translation_head = nn.Linear(hidden_dim//4, 3) # divide by 2 for ADD of rotation matrix, 4 for quaternion
        self.rotation_head = nn.Linear(hidden_dim//4, 4)  # divide by 2, 9 rotation matrix; //4, 4 per quaternion

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        #  FC layers
        x = self.fc_layers(features)

        # Predict translation and rotation
        translation = self.translation_head(x)
        rotation_flat = self.rotation_head(x)

        # Reshape rotation matrix in 3x3 shape
        batch_size = rotation_flat.size(0)
        rotation = rotation_flat.view(batch_size, 4) # 3, 3 for rotation matrix, 4 for quaternion

        return translation, rotation