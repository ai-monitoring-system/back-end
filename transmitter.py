import asyncio
import signal
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame
import os
import janus
import threading
import numpy as np

# Initialize Firebase
base_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(base_dir, "firebaseKey.json")
cred = credentials.Certificate(key_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Create an asyncio event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
stop_event = asyncio.Event()
frame_queue = None
frame_queue_ready = threading.Event()

# Define the ICE servers
ice_servers = [
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
    RTCIceServer(urls="stun:stun2.l.google.com:19302"),
]

# Define the VideoStreamTrack subclass
class FrameSenderTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, frame_queue):
        super().__init__()  # Don't forget this!
        self.frame_queue = frame_queue

    async def recv(self):
        frame = await self.frame_queue.async_q.get()
        frame.pts, frame.time_base = await self.next_timestamp()
        return frame

# Define the transmit_frame function
def transmit_frame(frame):
    #print("Transmitting frame...")
    frame_queue.sync_q.put(frame)
    #print("Frame transmitted")

# Define the send_frame function to be imported in process_frame
def send_frame(img):
    # Wait until frame_queue is initialized
    print("Sending frame...")
    if not frame_queue_ready.is_set():
        frame_queue_ready.wait()
    frame = VideoFrame.from_ndarray(img, format='bgr24')
    transmit_frame(frame)
    print("Frame sent")

async def test_stream_green_screen():
    """Function to send a green screen through the video stream."""
    # Wait until frame_queue is initialized
    if not frame_queue_ready.is_set():
        frame_queue_ready.wait()

    # Generate a green screen (BGR format, pure green)
    height, width = 480, 640  # Common video resolution
    green_screen = (0, 255, 0)  # Pure green in BGR
    img = cv2.rectangle(
        np.zeros((height, width, 3), dtype=np.uint8),
        (0, 0),
        (width, height),
        color=green_screen,
        thickness=-1
    )

    print("Streaming the green screen...")
    while not stop_event.is_set():
        # Create a VideoFrame from the green screen
        frame = VideoFrame.from_ndarray(img, format='bgr24')
        transmit_frame(frame)
        await asyncio.sleep(1)  # Simulate 1 FPS


async def run():
    global frame_queue
    # Initialize Janus queue inside the event loop
    frame_queue = janus.Queue()
    # Signal that frame_queue is ready
    frame_queue_ready.set()

    # Create a new call document in Firestore
    call_doc_ref = db.collection('calls').document()
    call_id = call_doc_ref.id
    print(f"Your call ID is: {call_id}")

    rtc_configuration = RTCConfiguration(iceServers=ice_servers)
    pc = RTCPeerConnection(rtc_configuration)

    # Add the video track to the RTCPeerConnection
    video_track = FrameSenderTrack(frame_queue)
    pc.addTrack(video_track)

    offer_candidates_ref = call_doc_ref.collection('offerCandidates')

    # Handle local ICE candidates
    @pc.on('icecandidate')
    def on_icecandidate(event):
        if event.candidate:
            candidate_dict = {
                'candidate': event.candidate.to_sdp(),
                'sdpMid': event.candidate.sdpMid,
                'sdpMLineIndex': event.candidate.sdpMLineIndex,
            }
            offer_candidates_ref.add(candidate_dict)

    # Handle connection state changes
    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == 'connected':
            print("ICE Connection established")
        if pc.connectionState in ['failed', 'disconnected', 'closed']:
            await pc.close()
            stop_event.set()

    # Create an offer and set the local description
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Write the offer to Firestore
    call_data = {'offer': {
        'type': pc.localDescription.type,
        'sdp': pc.localDescription.sdp
    }}
    call_doc_ref.set(call_data)

    answer_candidates_ref = call_doc_ref.collection('answerCandidates')

    # Listen for remote ICE candidates
    def on_answer_candidate_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == 'ADDED':
                data = change.document.to_dict()
                candidate = candidate_from_sdp(data['candidate'])
                candidate.sdpMid = data['sdpMid']
                candidate.sdpMLineIndex = int(data['sdpMLineIndex'])
                future = asyncio.run_coroutine_threadsafe(pc.addIceCandidate(candidate), loop)
                try:
                    future.result()
                except Exception as e:
                    print(f"Error adding ICE candidate: {e}")

    answer_candidates_ref.on_snapshot(on_answer_candidate_snapshot)

    # Listen for the answer from the web app
    def on_answer_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            data = doc.to_dict()
            if 'answer' in data:
                answer_desc = RTCSessionDescription(
                    sdp=data['answer']['sdp'],
                    type=data['answer']['type']
                )
                future = asyncio.run_coroutine_threadsafe(pc.setRemoteDescription(answer_desc), loop)
                try:
                    future.result()
                except Exception as e:
                    print(f"Error setting remote description: {e}")
                # Schedule unsubscribe on the main thread
                loop.call_soon_threadsafe(answer_listener.unsubscribe)

    answer_listener = call_doc_ref.on_snapshot(on_answer_snapshot)

    # Start the static image stream test
    await asyncio.gather(test_stream_green_screen(), stop_event.wait())

    # Close the Janus queue properly
    await frame_queue.async_q.join()
    frame_queue.close()
    await frame_queue.wait_closed()

def signal_handler():
    print("Signal received, shutting down")
    stop_event.set()

signal.signal(signal.SIGINT, lambda s, f: signal_handler())

if __name__ == '__main__':
    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
