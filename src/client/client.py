import asyncio
import websockets
from server.movement import perform_actions, mc
import numpy as np

URI = "wss://c617-133-16-42-18.ngrok-free.app" # Replace with your URI

async def client():
    try:
        async with websockets.connect(URI) as websocket:
            print("Connected to the server.")

            while True:
                message = await websocket.recv()
                # print("Received message:", message)

                joint_angles = np.array(eval(message)) 
                print("Parsed joint angles matrix (5x7):\n", joint_angles)
                perform_actions(mc, joint_angles.tolist())

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by the server.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(client())
