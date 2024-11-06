import asyncio
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay, MediaRecorder

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebaseKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Enter Call ID
call_id = input("Enter Call ID: ")

# Create a RTCPeerConnection
pc = RTCPeerConnection()
relay = MediaRelay()

# Clip length for recording (in seconds)
clip_length = 15

async def run():
    # Get the call document from Firestore
    call_doc_ref = db.collection('calls').document(call_id)
    call_doc = call_doc_ref.get()
    if not call_doc.exists:
        print("Call ID not found")
        return
    call_data = call_doc.to_dict()

    # Get the offer and set remote description
    offer = call_data.get('offer')
    if not offer:
        print("No offer in call document")
        return
    offer_desc = RTCSessionDescription(sdp=offer['sdp'], type=offer['type'])
    await pc.setRemoteDescription(offer_desc)

    # Create answer and set local description
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Update call document with the answer
    answer_dict = {
        'type': pc.localDescription.type,
        'sdp': pc.localDescription.sdp
    }
    call_data['answer'] = answer_dict
    call_doc_ref.set(call_data)

    # Set up ICE candidate collections
    answer_candidates_ref = call_doc_ref.collection('answerCandidates')
    offer_candidates_ref = call_doc_ref.collection('offerCandidates')

    # Handle local ICE candidates
    @pc.on('icecandidate')
    def on_icecandidate(event):
        if event.candidate:
            candidate_dict = {
                'candidate': event.candidate.__dict__['candidate'],
                'sdpMid': event.candidate.__dict__['sdpMid'],
                'sdpMLineIndex': event.candidate.__dict__['sdpMLineIndex'],
            }
            answer_candidates_ref.add(candidate_dict)

    # Listen for remote ICE candidates
    def on_offer_candidate_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == 'ADDED':
                data = change.document.to_dict()
                candidate = RTCIceCandidate(
                    sdpMid=data['sdpMid'],
                    sdpMLineIndex=data['sdpMLineIndex'],
                    candidate=data['candidate']
                )
                asyncio.ensure_future(pc.addIceCandidate(candidate))

    # Start listening to offer candidates
    offer_candidates_ref.on_snapshot(on_offer_candidate_snapshot)

    # Handle remote track
    @pc.on('track')
    def on_track(track):
        print("Track received: ", track.kind)
        if track.kind == 'video':
            local_video = relay.subscribe(track)
            # Start displaying video
            asyncio.ensure_future(display_video(local_video))
            # Start recording video
            asyncio.ensure_future(record_video(local_video))

    # Handle connection state changes
    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == 'failed' or pc.connectionState == 'disconnected':
            await pc.close()

    # Keep the program running
    await asyncio.Event().wait()

async def display_video(track):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        cv2.imshow('Remote Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    await pc.close()
    cv2.destroyAllWindows()
    print("Program terminated")
    exit(0)

async def record_video(track):
    index = 0
    while True:
        filename = f"recording_{index}.webm"
        recorder = MediaRecorder(filename)
        await recorder.start()

        # Record for clip_length seconds
        start_time = asyncio.get_event_loop().time()
        while True:
            frame = await track.recv()
            await recorder.write(frame)
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > clip_length:
                break

        await recorder.stop()
        index += 1
        print(f"Recording saved to {filename}")

# Run the main function
asyncio.run(run())
