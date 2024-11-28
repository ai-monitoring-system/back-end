import asyncio
import signal
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc.sdp import candidate_from_sdp
from display import process_frame

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(base_dir, "firebaseKey.json")
cred = credentials.Certificate(key_path)

#firebase_admin.initialize_app(cred)
db = firestore.client()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
relay = MediaRelay()
clip_length = 15
stop_event = asyncio.Event()


async def run():
    call_id = input("Enter Call ID: ")
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
        print(f"Local ICE candidate: {event.candidate}")
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
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == 'connected':
            print("ICE Connection established")
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
            print(f"Received remote ICE candidate: {change.document.to_dict()}")
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

    await stop_event.wait()


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
