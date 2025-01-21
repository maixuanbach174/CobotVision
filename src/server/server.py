import asyncio
import websockets

async def handler(websocket):
    print("Client connected")
    try:
        # Listen for messages
        async for message in websocket:
            print("Received message from client:", message)
            # Send a response back to the client
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print("Error:", e)
    finally:
        print("Client disconnected")

async def main():
    print("Starting WebSocket server...")
    async with websockets.serve(handler, "0.0.0.0", 8080):
        print("WebSocket server running on ws://0.0.0.0:8080")
        await asyncio.Future()  # Keeps the server running forever

if __name__ == "__main__":
    asyncio.run(main())
