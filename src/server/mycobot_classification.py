import cv2
import torch
from torchvision import transforms
import numpy as np
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

# Define the model
class ImageToJointAngles(nn.Module):
    def __init__(self):
        super(ImageToJointAngles, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 30)  # 5x6 = 30 joint angles
        )
        
    def forward(self, x):
        features = self.resnet(x)
        joint_angles = self.fc(features)
        joint_angles = joint_angles.view(-1, 5, 6)
        return joint_angles

# Load the model
model = ImageToJointAngles()
model.load_state_dict(torch.load("models/modelV0.pth", map_location="cpu",weights_only=True))
model.eval()

# Function to preprocess and predict
def predict(image: torch.Tensor, model: nn.Module) -> torch.Tensor:
    append_list = [-1, 1, -1, 0, -1]
    append_column = np.array(append_list).reshape(-1, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(224, 224))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.inference_mode():
        joint_angles = model(image_tensor)
    joint_angles = joint_angles.cpu().numpy().flatten()
    joint_angles = np.round(joint_angles.reshape(5, 6) * 180.0)
    joint_angles = np.hstack((joint_angles, append_column))
    return joint_angles
