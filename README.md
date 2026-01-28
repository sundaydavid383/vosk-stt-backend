# Vosk Backend - Free Demo

This is a Vosk-only backend for short user audio clips.  
Frontend handles recording; backend receives audio (base64), transcribes, and sends results back.

**Usage**:
- Connect via WebSocket to ws://your-domain:8765
- Send messages:
  1. `{ "type": "start", "user_id": "123", "sessionId": "abc" }`
  2. `{ "type": "audio_chunk", "audio": "<base64 audio>" }`
  3. `{ "type": "end" }`
