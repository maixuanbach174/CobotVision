from websocket import create_connection

try:
    # Replace with your WebSocket server URL
    ws_url = "wss://43d1-133-16-42-18.ngrok-free.app"
    
    # Attempt to create a connection
    ws = create_connection(ws_url)
    print("Connected to WebSocket successfully!")

    # Optional: Send a test message
    ws.send("Hello, WebSocket server!")
    response = ws.recv()
    print("Received response:", response)

    # Close the connection
    ws.close()
except Exception as e:
    print("Failed to connect to WebSocket:", e)
