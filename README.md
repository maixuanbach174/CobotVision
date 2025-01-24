# **Deep Learning-Based Robot Arm for Real-Time Object Detection and Classification**

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Documentation and Demo](#documentation-and-demo)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [System Design](#system-design) 
7. [How to Run](#how-to-run)
8. [Troubleshooting](#troubleshooting)
9. [Acknowledgements](#acknowledgements)
10. [License](#license)

---

## **1. Introduction**
This project uses the MyCobot 280 Pi robotic arm for classifying and sorting objects into designated boxes based on their colors. It integrates:
- **Deep learning** for joint angle prediction.
- **WebSocket communication** for real-time control.
- A public server setup for seamless robot-laptop interaction.

### **Demo**
![Project Demo GIF](#link-to-gif-or-screenshot)  

---

## **2. Features**
- **Real-Time Detection and Classification**: The robot detects colored cubes and classifies them into appropriate boxes.
- **Custom Dataset Creation**: Supports dataset with structured image and joint angle mapping.
- **WebSocket Communication**: Enables smooth communication between the laptop and MyCobot.
- **Easy Integration**: Designed for anyone looking to expand robotic capabilities.

---

## **6. Documentation and Demo**
- **Documentation**: [Access Full Documentation](#link-to-documentation)
- **Demo Video**: [Watch the Demo](#link-to-demo-video)
- **Project Repository**: [GitHub Repository](#link-to-repository)

---

## **3. Dataset**
### **Structure**
The dataset consists of folders representing actions. Each folder contains:
- `image.png`: Input image (190x190).
- `joint_angles.json`: Corresponding joint angles matrix (5x7).

Example:
```
dataset/
  ├── action_1/
  │   ├── image.png
  │   └── joint_angles.json
  ├── action_2/
  │   ├── image.png
  │   └── joint_angles.json
  └── ...
```

### **JSON Structure**
- 5x7 matrix representing joint angles.
- Specific rows/columns:
  - Row 4, Column 1: Color (e.g., Red: 93, Yellow: 53, Green: -10, Blue: -23).
  - Rows 1 & 2: Object position.

![Dataset Example](#link-to-dataset-image)

---

## **4. Deep Learning Model**
### **Model Overview**
The deep learning model predicts joint angles based on input images. 
- **Input**: RGB image (224x224).
- **Output**: 5x6 matrix representing joint angles.

### **Model Diagram**
![Model Structure](#link-to-model-structure-image)

### **Training Details**
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam.
- **Key Hyperparameters**: Batch size: 32, Learning rate: 0.001.

---

## **5. System Design**
### **WebSocket Communication**
The system uses WebSocket to connect the laptop and MyCobot via a public server.

### **System Flow**
1. **Laptop**: Captures an image, predicts joint angles, and sends data via WebSocket.
2. **Server**: Processes and forwards commands.
3. **MyCobot**: Executes actions based on received joint angles.

![System Diagram](#link-to-server-design-image)

---

## **7. How to Run**
### **Prerequisites**
- Python 3.8+
- MyCobot 280 Pi with Raspberry Pi 4 OS.

### **Setup Instructions**
1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```
2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Server**:  
   ```bash
   python server.py
   ```
4. **Run the Client**:  
   ```bash
   python client.py
   ```
5. **Connect MyCobot**: Ensure MyCobot is powered on and connected.
6. **Start the System**: Begin real-time classification and sorting.

---

## **8. Troubleshooting**
- **Issue**: MyCobot not responding.
  - **Solution**: Check USB connection and ensure the robot is powered on.
- **Issue**: WebSocket connection error.
  - **Solution**: Verify server is running and firewall settings allow communication.
- **Issue**: Incorrect classification results.
  - **Solution**: Ensure the dataset and model weights are properly loaded.

---

## **9. Acknowledgements**
- [Elephant Robotics](https://www.elephantrobotics.com) for MyCobot 280 Pi.
- Open-source libraries and frameworks used in the project.

---

## **10. License**
This project is licensed under the MIT License. See the [LICENSE](#link-to-license-file) file for details.

