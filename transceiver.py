import asyncio
import signal
import os
import cv2
import numpy as np

import firebase_admin
from firebase_admin import credentials, firestore

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame

from display import process_frame

stop_event = asyncio.Event()
processed_video_track = None
MAIN_LOOP = None

def run_yolo_inference(img: np.ndarray) -> np.ndarray:
    return process_frame(img)

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
        self.frame_queue.put_nowait(frame_ndarray)

async def main():
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_event_loop()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(base_dir, "firebaseKey.json")

    # Initialize Firebase only once
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    pc_in = RTCPeerConnection()

    # Prompt for the inbound call ID (already created by Web App A)
    call_id_in = input("Inbound call ID: ")
    call_doc_ref_in = db.collection("calls").document(call_id_in)
    call_doc_in = call_doc_ref_in.get()
    if not call_doc_in.exists:
        print(f"Inbound call ID {call_id_in} not found in Firestore.")
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

    # Create a brand new Firestore doc for the outbound call
    call_doc_ref_out = db.collection("calls").document()
    call_id_out = call_doc_ref_out.id
    print(f"Outbound Call ID: {call_id_out}")

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

async def handle_inbound_video(track, pc_in):
    global processed_video_track

    while True:
        try:
            frame = await track.recv()
        except Exception as e:
            print("Error receiving frame:", e)
            break

        img = frame.to_ndarray(format="bgr24")
        processed_img = run_yolo_inference(img)
        processed_video_track.push_frame(processed_img)

    # Cleanup if the loop ends
    await pc_in.close()
    stop_event.set()

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
