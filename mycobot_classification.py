import cv2
import torch
from torchvision import transforms
import numpy as np
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from pymycobot.mycobot import MyCobot
import time
from movement import perform_actions

mc = MyCobot('/dev/ttyAMA0', 1000000)

# Load the pre-trained model
class ImageToJointAngles(nn.Module):
    def __init__(self):
        super(ImageToJointAngles, self).__init__()
        # Backbone: Use a pretrained ResNet18 for feature extraction
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Fully connected layers to predict joint angles
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18 outputs 512 features
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 30)  # 5x6 = 30 joint angles
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.resnet(x)
        
        # Fully connected layers
        joint_angles = self.fc(features)
        
        # Reshape output to (batch_size, 5, 6)
        joint_angles = joint_angles.view(-1, 5, 6)
        
        return joint_angles


model = ImageToJointAngles()
model.load_state_dict(torch.load("model/modelV0.pth", weights_only=True))
model.eval()

def predict(image: torch.Tensor, model: nn.Module) -> torch.Tensor:
    append_list = [-1, 1, -1, 0, -1]
    append_column = np.array(append_list).reshape(-1, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
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

# Define image preprocessing pipeline
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = frame[254:444, 537:727]
image = frame

while(True):
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        joint_angles = predict(image, model)
        perform_actions(mc, joint_angles)
        cv2.destroyAllWindows()
        break
cap.release()

sudo apt update
sudo apt install -y build-essential libssl-dev libffi-dev zlib1g-dev libncurses5-dev libnss3-dev libsqlite3-dev libreadline-dev libbz2-dev

wget https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz
tar -xvzf Python-3.11.5.tgz
cd Python-3.11.5

./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall  # Use altinstall to avoid overwriting the default `python3`



