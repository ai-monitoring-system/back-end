# Backend Setup

This backend is responsible for real-time video streaming, AI-based person detection using YOLO, and sending push notifications via Firebase Cloud Messaging.

It is one part of the full project. You must start this backend **before** running the frontend for streaming or viewing. Refer to the main project `README.md` for full setup instructions.

---

## Getting Started

### 1. Install Python Dependencies

Make sure Python is installed. Then install the required libraries:

```bash
pip install -r requirements.txt
```

### 2. Start the Signaling Server

We use Node.js for WebSocket signaling:

```bash
node server.js
```

> Note: Make sure Node.js is installed on your machine. [Download it here](https://nodejs.org/).

---

## What's in this folder

- `transceiver.py`: handles WebRTC video streams, person detection, and sending FCM push notifications
- `firebaseKey.json`: your Firebase Admin SDK credentials
- `requirements.txt`: Python dependencies for the backend
- `server.js`: the WebSocket signaling server

---

After this server is running, you can open the frontend to start the camera stream and view it live with notifications.
