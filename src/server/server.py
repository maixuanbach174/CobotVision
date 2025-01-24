import asyncio
import websockets
from mycobot_classification import predict, model
import cv2

async def handler(websocket):
    print("Client connected")
    try:
        cap = cv2.VideoCapture(0)  # Initialize the camera
        print("Press 's' to capture and process an image or 'q' to quit.")

        while True:
            ret, frame = cap.read()
            cropped_frame = frame[383:667, 796:1080]
            if not ret:
                print("Failed to capture frame.")
                break

            # Display the frame in real-time
            cv2.imshow("Captured Frame", cropped_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Press 's' to capture and process an image
                print("Capturing and processing image...")
                # Predict joint angles using the model
                joint_angles = predict(cropped_frame, model)
                print("Predicted Joint Angles:\n", joint_angles)

                # Send joint angles to the client
                await websocket.send(str(joint_angles.tolist()))
                print("Joint angles sent to the client.")

            elif key == ord('q'):  # Press 'q' to quit
                print("Exiting...")
                break

        cap.release()  # Release the camera resource
        cv2.destroyAllWindows()  # Close OpenCV windows

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print("Error:", e)
    finally:
        print("Client disconnected")
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

# Start the WebSocket server
async def main():
    print("Starting WebSocket server...")
    async with websockets.serve(handler, "0.0.0.0", 8080):
        print("WebSocket server running on ws://0.0.0.0:8080")
        await asyncio.Future()  # Keeps the server running

if __name__ == "__main__":
    asyncio.run(main())
