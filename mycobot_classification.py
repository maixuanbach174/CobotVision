import cv2
import torch
from torchvision import transforms
import numpy as np
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

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
model.load_state_dict(torch.load("model/modelV0.pth"))
model.eval()

# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Model's input size
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the region of interest (ROI)
ROI_TOP_LEFT = (100, 100)
ROI_BOTTOM_RIGHT = (400, 400)

# Capture real-time video feed
cap = cv2.VideoCapture(0)  # Replace with your camera's ID
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the region of interest (ROI) on the frame
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
    roi = frame[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]

    # Preprocess the ROI for model input
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_tensor = transform(roi_rgb).unsqueeze(0)  # Add batch dimension

    # Predict using the model
    with torch.no_grad():
        predicted_angles = model(roi_tensor).squeeze(0)  # Shape: (5, 7)

    # Extract gripper state and joint angles
    gripper_state = predicted_angles[4, 0].item()  # Example: gripper control value
    joint_angles = predicted_angles[:5, :6].flatten().tolist()  # Extract joint angles

    # If the ROI is blank (no object), do nothing
    if np.sum(roi) < 1000:  # Example threshold for detecting blank ROI
        print("No object detected. MyCobot is idle.")
        continue

    # If object is detected, execute gripping motion
    print("Object detected! Executing motion.")
    print(joint_angles)
    # Display the real-time frame
    cv2.imshow("Real-Time Gripping", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
