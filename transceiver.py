import asyncio
import signal
import os
import cv2
import numpy as np
import time  # Add this import for tracking time

import firebase_admin
from firebase_admin import credentials, firestore, messaging

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame

from display import process_frame

stop_event = asyncio.Event()
processed_video_track = None
MAIN_LOOP = None

###############################################################################
# GLOBALS
###############################################################################
user_id = None
db = None
last_notification_time = 0  # Track the last time a notification was sent
NOTIFICATION_COOLDOWN = 5  # Cooldown period in seconds

###############################################################################
# YOLO & Firestore Helpers
###############################################################################
def run_yolo_inference(img: np.ndarray):
    """
    run_yolo_inference calls process_frame(img),
    which returns (processed_img, found_person).
    """
    return process_frame(img)  # e.g. returns (img_with_boxes, bool_found_person)

def handle_person_detected():
    """
    Fetches the user's FCM tokens from Firestore and sends an FCM push notification
    directly from Python instead of using Firebase Cloud Functions.
    """
    global user_id, db, last_notification_time

    # Check if the cooldown period has passed
    current_time = time.time()
    if current_time - last_notification_time < NOTIFICATION_COOLDOWN:
        return

    if not user_id or not db:
        print("Warning: user_id or db not set - cannot send notification.")
        return

    # Fetch FCM tokens from Firestore
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        print(f"No user found for ID: {user_id}")
        return

    user_data = user_doc.to_dict()
    tokens = user_data.get("fcmTokens", [])

    if not tokens:
        print(f"No FCM tokens found for user ID: {user_id}")
        return

    # Create the push notification message
    notification = messaging.Notification(
        title="🚨 Person Detected!",
        body="A person was detected on camera. Check your feed!",
    )

    # Send FCM notification to each token individually
    for token in tokens:
        message = messaging.Message(
            notification=notification,
            token=token,
        )
        try:
            response = messaging.send(message)
            print(f"FCM Message sent to {token}: {response}")
        except Exception as e:
            print(f"Error sending FCM notification to {token}: {e}")

    # Log detection in Firestore (optional)
    data = {
        "userId": user_id,
        "type": "personDetected",
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    db.collection("notifications").add(data)
    print("Wrote personDetected event to Firestore:", data)

    # Update the last notification time
    last_notification_time = current_time

###############################################################################
# Video Stream Track
###############################################################################
class ProcessedVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.frame_queue = asyncio.Queue()

    async def recv(self):
        frame_ndarray = await self.frame_queue.get()

        av_frame = VideoFrame.from_ndarray(frame_ndarray, format="bgr24")
        av_frame.pts, av_frame.time_base = await self.next_timestamp()
        return av_frame

    def push_frame(self, frame_ndarray: np.ndarray):
        if self.frame_queue.qsize() >= 60:
            self.frame_queue.get_nowait()
        self.frame_queue.put_nowait(frame_ndarray)

###############################################################################
# Main Entry
###############################################################################
async def main():
    global MAIN_LOOP, user_id, db
    MAIN_LOOP = asyncio.get_event_loop()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(base_dir, "firebaseKey.json")

    # Initialize Firebase only once
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)

    # Set the global db
    db = firestore.client()

    # Prompt for the user ID once, store it in the global user_id
    user_id = input("User ID: ")
    print(f"Using user_id = {user_id}")

    pc_in = RTCPeerConnection()

    # Prompt for the user ID once
    user_id = input("User ID: ")
    call_doc_ref_in = db.collection("calls").document(user_id)

    # Clear any previous call session data
    call_doc_ref_in.set({}, merge=True)  # Clears existing fields but keeps the document

    call_doc_in = call_doc_ref_in.get()
    if not call_doc_in.exists:
        print(f"Inbound call ID {user_id} not found in Firestore.")
        return

    call_data_in = call_doc_in.to_dict()
    offer_in = call_data_in.get("offer")
    if not offer_in:
        print("No 'offer' found in inbound call doc.")
        return

    inbound_offer_desc = RTCSessionDescription(
        sdp=offer_in["sdp"],
        type=offer_in["type"]
    )

    @pc_in.on("icecandidate")
    def on_in_icecandidate(event):
        if event.candidate is not None:
            print("pc_in local ICE candidate:", event.candidate)
            candidate_dict = {
                "candidate": event.candidate.to_sdp(),
                "sdpMid": event.candidate.sdpMid,
                "sdpMLineIndex": event.candidate.sdpMLineIndex,
            }
            call_doc_ref_in.collection("answerCandidates").add(candidate_dict)

    @pc_in.on("connectionstatechange")
    async def on_in_state_change():
        print("pc_in state:", pc_in.connectionState)
        if pc_in.connectionState in ["failed", "disconnected", "closed"]:
            await pc_in.close()
            stop_event.set()

    @pc_in.on("track")
    def on_in_track(track):
        #print(f"pc_in got track: {track.kind}")
        if track.kind == "video":
            # Launch a coroutine to handle inbound frames
            asyncio.ensure_future(handle_inbound_video(track, pc_in))

    def on_offer_candidate_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == "ADDED":
                data = change.document.to_dict()
                candidate_sdp = data["candidate"]
                candidate = candidate_from_sdp(candidate_sdp)
                candidate.sdpMid = data["sdpMid"]
                candidate.sdpMLineIndex = int(data["sdpMLineIndex"])

                # Schedule the coroutine on the MAIN_LOOP
                future = asyncio.run_coroutine_threadsafe(
                    pc_in.addIceCandidate(candidate),
                    MAIN_LOOP
                )
                try:
                    future.result()
                except Exception as e:
                    print("Error adding inbound ICE candidate:", e)

    # Listen for inbound ICE from the "offerCandidates" sub-collection
    call_doc_ref_in.collection("offerCandidates").on_snapshot(on_offer_candidate_snapshot)

    # Set remote description to inbound offer, then create & save local answer
    await pc_in.setRemoteDescription(inbound_offer_desc)
    answer_in = await pc_in.createAnswer()
    await pc_in.setLocalDescription(answer_in)

    call_data_in["answer"] = {
        "type": pc_in.localDescription.type,
        "sdp": pc_in.localDescription.sdp
    }
    call_doc_ref_in.set(call_data_in)

    pc_out = RTCPeerConnection()

    # Use the same Firestore doc for the outbound call
    call_doc_ref_out = call_doc_ref_in

    @pc_out.on("icecandidate")
    def on_out_icecandidate(event):
        if event.candidate is not None:
            print("pc_out local ICE candidate:", event.candidate)
            candidate_dict = {
                "candidate": event.candidate.to_sdp(),
                "sdpMid": event.candidate.sdpMid,
                "sdpMLineIndex": event.candidate.sdpMLineIndex,
            }
            call_doc_ref_out.collection("offerCandidates").add(candidate_dict)

    @pc_out.on("connectionstatechange")
    async def on_out_state_change():
        print("pc_out state:", pc_out.connectionState)
        if pc_out.connectionState in ["failed", "disconnected", "closed"]:
            await pc_out.close()
            stop_event.set()

    global processed_video_track
    processed_video_track = ProcessedVideoStreamTrack()
    pc_out.addTrack(processed_video_track)

    # Create an offer for Web App B
    offer_out = await pc_out.createOffer()
    await pc_out.setLocalDescription(offer_out)

    # Save the outbound offer in Firestore
    call_doc_ref_out.set({
        "offer": {
            "type": pc_out.localDescription.type,
            "sdp": pc_out.localDescription.sdp
        }
    })

    # Watch for the answer from Web App B
    def on_out_call_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            data = doc.to_dict()
            if "answer" in data:
                ans = data["answer"]
                answer_desc = RTCSessionDescription(
                    sdp=ans["sdp"],
                    type=ans["type"]
                )
                future = asyncio.run_coroutine_threadsafe(
                    pc_out.setRemoteDescription(answer_desc),
                    MAIN_LOOP
                )
                try:
                    future.result()
                except Exception as e:
                    print("Error setting outbound remote description:", e)

    call_doc_ref_out.on_snapshot(on_out_call_snapshot)

    def on_out_answer_candidate_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == "ADDED":
                data = change.document.to_dict()
                candidate_sdp = data["candidate"]
                candidate = candidate_from_sdp(candidate_sdp)
                candidate.sdpMid = data["sdpMid"]
                candidate.sdpMLineIndex = int(data["sdpMLineIndex"])

                future = asyncio.run_coroutine_threadsafe(
                    pc_out.addIceCandidate(candidate),
                    MAIN_LOOP
                )
                try:
                    future.result()
                except Exception as e:
                    print("Error adding outbound ICE candidate:", e)

    call_doc_ref_out.collection("answerCandidates").on_snapshot(on_out_answer_candidate_snapshot)

    await stop_event.wait()
    print("Stop event triggered - shutting down...")

    # Clean up
    await pc_in.close()
    await pc_out.close()

    # Clean up Firestore doc & sub-collections
    try:
        # Delete sub-collections first (offerCandidates, answerCandidates)
        offer_candidates = call_doc_ref_in.collection("offerCandidates").stream()
        answer_candidates = call_doc_ref_in.collection("answerCandidates").stream()
        
        for candidate in offer_candidates:
            candidate.reference.delete()
        
        for candidate in answer_candidates:
            candidate.reference.delete()

        # Now delete the main doc
        call_doc_ref_in.delete()
    except Exception as e:
        print(f"Error deleting old call data: {e}")

###############################################################################
# Handling inbound video frames
###############################################################################
async def handle_inbound_video(track, pc_in):
    global processed_video_track

    while True:
        try:
            frame = await track.recv()
        except Exception as e:
            print("Error receiving frame:", e)
            break

        img = frame.to_ndarray(format="bgr24")

        # run_yolo_inference returns (processed_img, found_person)
        processed_img, found_person = run_yolo_inference(img)

        # If a person was detected, write the "notifications" doc
        if found_person:
            handle_person_detected()

        # Continue streaming the processed frame out
        processed_video_track.push_frame(processed_img)

    # Cleanup if the loop ends
    await pc_in.close()
    stop_event.set()

###############################################################################
# Signal Handling for graceful shutdown
###############################################################################
def signal_handler(sig, frame):
    print("Signal received, shutting down...")
    stop_event.set()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()