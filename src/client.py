import asyncio
import websockets
import numpy as np

async def client():
    uri = "ws://c617-133-16-42-18.ngrok-free.app"

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to the server.")

            while True:
                message = await websocket.recv()
                print("Received message:", message)

                joint_angles = np.array(eval(message)) 
                print("Parsed joint angles matrix (5x7):\n", joint_angles)

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by the server.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(client())
