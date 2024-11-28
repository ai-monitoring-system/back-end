import asyncio
import signal
import cv2
import os
import threading
import firebase_admin
from firebase_admin import credentials, firestore
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame
import janus
from ultralytics import YOLO  # Assuming you're using ultralytics YOLO

# Initialize Firebase
base_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(base_dir, "firebaseKey.json")
cred = credentials.Certificate(key_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Event Loop
loop = asyncio.get_event_loop()

# Global Variables
relay = MediaRelay()
stop_event = asyncio.Event()

# Declare frame_queue as a global variable
frame_queue = None

# Threading event to signal when frame_queue is ready
frame_queue_ready = threading.Event()

# Load YOLO model (update the path to your model if necessary)
model_path = os.path.join(base_dir, "models", "yolo11n.pt")
model = YOLO(model_path, verbose=False)

# Define the VideoStreamTrack subclass for Transmitter
class FrameSenderTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue

    async def recv(self):
        frame = await self.frame_queue.async_q.get()
        frame.pts, frame.time_base = await self.next_timestamp()
        return frame

def transmit_frame(frame):
    frame_queue.sync_q.put(frame)

def send_frame(img):
    # Wait until frame_queue is initialized
    if not frame_queue_ready.is_set():
        frame_queue_ready.wait()
    frame = VideoFrame.from_ndarray(img, format='bgr24')
    transmit_frame(frame)

def process_frame(img):
    try:
        # Your processing code here (e.g., YOLO detection)
        results = model(img)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                color = (0, int(conf * 255), int((1 - conf) * 255))
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        send_frame(img)

        # Display the image if needed
        cv2.imshow('Processed Video', img)
        key = cv2.waitKey(1)

        if key == 27 or cv2.getWindowProperty('Processed Video', cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True
    except Exception as e:
        print(f"Exception in process_frame: {e}")
        return False

async def run():
    global frame_queue
    # Initialize Janus queue inside the event loop
    frame_queue = janus.Queue()
    # Signal that frame_queue is ready
    frame_queue_ready.set()

    # --- Receiver Setup ---
    call_id = input("Enter Call ID for Receiving Video: ")
    pc = RTCPeerConnection()

    call_doc_ref = db.collection('calls').document(call_id)
    call_doc = call_doc_ref.get()
    if not call_doc.exists:
        print("Call ID not found")
        stop_event.set()
        return
    call_data = call_doc.to_dict()

    offer = call_data.get('offer')
    if not offer:
        print("No offer in call document")
        stop_event.set()
        return
    offer_desc = RTCSessionDescription(sdp=offer['sdp'], type=offer['type'])

    @pc.on('icecandidate')
    def on_icecandidate(event):
        if event.candidate:
            candidate_dict = {
                'candidate': event.candidate.to_sdp(),
                'sdpMid': event.candidate.sdpMid,
                'sdpMLineIndex': event.candidate.sdpMLineIndex,
            }
            answer_candidates_ref.add(candidate_dict)

    @pc.on('track')
    def on_track(track):
        print(f"Track received: {track.kind}")
        if track.kind == 'video':
            local_video = relay.subscribe(track)
            print("Starting display_video coroutine")
            asyncio.ensure_future(display_video(local_video, pc))

    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        print("Receiver Connection state is %s" % pc.connectionState)
        if pc.connectionState == 'connected':
            print("Receiver ICE Connection established")
        if pc.connectionState in ['failed', 'disconnected', 'closed']:
            await pc.close()
            stop_event.set()

    await pc.setRemoteDescription(offer_desc)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    answer_dict = {
        'type': pc.localDescription.type,
        'sdp': pc.localDescription.sdp
    }
    call_data['answer'] = answer_dict
    call_doc_ref.set(call_data)

    answer_candidates_ref = call_doc_ref.collection('answerCandidates')
    offer_candidates_ref = call_doc_ref.collection('offerCandidates')

    def on_offer_candidate_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == 'ADDED':
                data = change.document.to_dict()
                candidate_sdp = data['candidate']
                candidate = candidate_from_sdp(candidate_sdp)
                candidate.sdpMid = data['sdpMid']
                candidate.sdpMLineIndex = int(data['sdpMLineIndex'])
                future = asyncio.run_coroutine_threadsafe(pc.addIceCandidate(candidate), loop)
                try:
                    future.result()
                except Exception as e:
                    print(f"Error adding ICE candidate: {e}")

    offer_candidates_ref.on_snapshot(on_offer_candidate_snapshot)

    # --- Transmitter Setup ---
    transmit_call_doc_ref = db.collection('calls').document()
    transmit_call_id = transmit_call_doc_ref.id
    print(f"Your Call ID for Transmitting Processed Video is: {transmit_call_id}")

    transmitter_pc = RTCPeerConnection()

    # Add the video track to the RTCPeerConnection
    video_track = FrameSenderTrack(frame_queue)
    transmitter_pc.addTrack(video_track)

    transmit_offer_candidates_ref = transmit_call_doc_ref.collection('offerCandidates')

    # Handle local ICE candidates for transmitter
    @transmitter_pc.on('icecandidate')
    def on_transmitter_icecandidate(event):
        if event.candidate:
            candidate_dict = {
                'candidate': event.candidate.to_sdp(),
                'sdpMid': event.candidate.sdpMid,
                'sdpMLineIndex': event.candidate.sdpMLineIndex,
            }
            transmit_offer_candidates_ref.add(candidate_dict)

    # Handle connection state changes for transmitter
    @transmitter_pc.on('connectionstatechange')
    async def on_transmitter_connectionstatechange():
        print(f"Transmitter Connection state is {transmitter_pc.connectionState}")
        if transmitter_pc.connectionState == 'connected':
            print("Transmitter ICE Connection established")
        if transmitter_pc.connectionState in ['failed', 'disconnected', 'closed']:
            await transmitter_pc.close()
            stop_event.set()

    # Create an offer and set the local description for transmitter
    transmit_offer = await transmitter_pc.createOffer()
    await transmitter_pc.setLocalDescription(transmit_offer)

    # Write the offer to Firestore
    transmit_call_data = {'offer': {
        'type': transmitter_pc.localDescription.type,
        'sdp': transmitter_pc.localDescription.sdp
    }}
    transmit_call_doc_ref.set(transmit_call_data)

    transmit_answer_candidates_ref = transmit_call_doc_ref.collection('answerCandidates')

    # Listen for remote ICE candidates for transmitter
    def on_transmitter_answer_candidate_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == 'ADDED':
                data = change.document.to_dict()
                candidate = candidate_from_sdp(data['candidate'])
                candidate.sdpMid = data['sdpMid']
                candidate.sdpMLineIndex = int(data['sdpMLineIndex'])
                future = asyncio.run_coroutine_threadsafe(transmitter_pc.addIceCandidate(candidate), loop)
                try:
                    future.result()
                except Exception as e:
                    print(f"Error adding ICE candidate: {e}")

    transmit_answer_candidates_ref.on_snapshot(on_transmitter_answer_candidate_snapshot)

    # Listen for the answer from the web app for transmitter
    def on_transmitter_answer_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            data = doc.to_dict()
            if 'answer' in data:
                answer_desc = RTCSessionDescription(
                    sdp=data['answer']['sdp'],
                    type=data['answer']['type']
                )
                future = asyncio.run_coroutine_threadsafe(transmitter_pc.setRemoteDescription(answer_desc), loop)
                try:
                    future.result()
                except Exception as e:
                    print(f"Error setting remote description: {e}")
                # Schedule unsubscribe on the main thread
                loop.call_soon_threadsafe(transmitter_answer_listener.unsubscribe)

    transmitter_answer_listener = transmit_call_doc_ref.on_snapshot(on_transmitter_answer_snapshot)

    # Wait until the connection is closed
    await stop_event.wait()

    # Close the Janus queue properly
    await frame_queue.async_q.join()
    frame_queue.close()
    await frame_queue.wait_closed()

async def display_video(track, pc):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        should_continue = process_frame(img)
        if not should_continue:
            print("Exiting display_video coroutine")
            break

    await pc.close()
    cv2.destroyAllWindows()
    print("Program terminated")
    stop_event.set()

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
