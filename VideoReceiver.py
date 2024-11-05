import asyncio
import websockets
import av
import numpy as np
import cv2
import io

URI = "ws://localhost:8080"

async def video_receiver(websocket, path):
    print("WebSocket connection established")
    
    buffer = io.BytesIO()

    try:
        async for message in websocket:
            buffer.write(message)
            buffer.seek(0)
            
            try:
                container = av.open(buffer, format='webm')
                
                for frame in container.decode(video=0):
                    # Convert the frame to an OpenCV image format (BGR)
                    img = frame.to_ndarray(format="bgr24")
                    cv2.imshow("Video Stream", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            except av.AVError as e:
                print("Error decoding video:", e)
            
            buffer.seek(0)
            buffer.truncate()

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    
    finally:
        cv2.destroyAllWindows()

async def main():
    async with websockets.serve(video_receiver, "localhost", 8080):
        print("WebSocket server started at ws://localhost:8080")
        await asyncio.Future()

asyncio.run(main())
