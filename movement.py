import json
from pymycobot.mycobot import MyCobot
import time

# Initialize MyCobot connection (replace with your port)
mc = MyCobot('/dev/ttyAMA0', 1000000)

# Load actions from JSON file
def load_actions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data["actions"]

# Perform actions
def perform_actions(mc, actions):
    for idx, action in enumerate(actions, start=1):
        angles = action[:-1]  # Extract joint angles
        grip_status = action[-1]  # Extract grip status

        print(f"Performing Action {idx}:")
        print(f"  Joint Angles: {angles}")
        print(f"  Grip Status: {'Closed' if grip_status == 1 else 'Open'}")

        # Move to the joint angles
        mc.send_angles(angles, 50)  # Adjust speed as needed
        time.sleep(1)  # Allow movement to complete

        # Grip control
        
    
    print("All actions completed!")

# File path
file_path = "dataset/action_312/joint_angles.json"


# Main
if __name__ == "__main__":
    actions = load_actions(file_path)
    perform_actions(mc, actions)