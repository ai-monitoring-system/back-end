import asyncio
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaRecorder
from aiortc.sdp import candidate_from_sdp
import threading
import signal

cred = credentials.Certificate("opencv-visualizer/firebaseKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
pc = RTCPeerConnection()
relay = MediaRelay()
clip_length = 15
stop_event = asyncio.Event()

async def run():
    call_id = input("Enter Call ID: ")

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

    @pc.on('icecandidate')
    def on_icecandidate(event):
        if event.candidate:
            candidate_dict = {
                'candidate': event.candidate.to_sdp(),
                'sdpMid': event.candidate.sdpMid,
                'sdpMLineIndex': event.candidate.sdpMLineIndex,
            }
            answer_candidates_ref.add(candidate_dict)

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

    @pc.on('track')
    def on_track(track):
        print("Track received: ", track.kind)
        if track.kind == 'video':
            local_video = relay.subscribe(track)
            print("Starting display_video coroutine")
            asyncio.ensure_future(display_video(local_video))
            print("Starting record_video coroutine")
            asyncio.ensure_future(record_video(local_video))

    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState in ['failed', 'disconnected', 'closed']:
            await pc.close()
            stop_event.set()

    await stop_event.wait()

async def display_video(track):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        cv2.imshow('Remote Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting display_video coroutine")
            break

    await pc.close()
    cv2.destroyAllWindows()
    print("Program terminated")
    stop_event.set()

async def record_video(track):
    index = 0
    while not stop_event.is_set():
        filename = f"recording_{index}.webm"
        recorder = MediaRecorder(filename)
        await recorder.start()

        start_time = asyncio.get_event_loop().time()
        while True:
            frame = await track.recv()
            await recorder.write(frame)
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > clip_length or stop_event.is_set():
                break

        await recorder.stop()
        index += 1
        print(f"Recording saved to {filename}")

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
