# Cymbal Assist — Real-time Call Center Agent Assist

Real-time call-center agent-assist dashboard that transcribes customer speech live,
retrieves context from a Vertex AI RAG corpus, and streams RAG-grounded Gemini responses —
all in a WhatsApp-style chat UI the agent can read while on the call.

## Architecture

```
Browser Mic ──► AudioWorklet (20ms PCM-16 chunks)
                    │
                WebSocket (binary)
                    │
               FastAPI Server (Cloud Run)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Streaming STT             Gemini 2.5 Flash
  (Chirp 3, V2 API)        + RAG Tool (google.genai SDK)
  interim + final           streaming chunks
  transcripts               via send_message_stream
        │                       │
        └───────────┬───────────┘
                    ▼
             WebSocket (JSON)
                    │
           Agent Dashboard
        ┌──────────┴──────────┐
        │ 🎤 Customer         │ 🤖 AI Assistant
        │ (live transcript)   │ (streamed, RAG-grounded)
        │                     │ ⏱ latency badge
        └─────────────────────┘
```

**Pipeline** (no TTS — agent reads responses on screen):

```
Audio (PCM-16, 16kHz) → Streaming STT (Chirp 3) → Final transcript
    → Gemini 2.5 Flash (with RAG tool, streaming) → Agent dashboard
```

## Key Features

- **Streaming STT** — Chirp 3 model via Cloud Speech V2 with interim results for real-time display
- **Auto language detection** — Chirp 3 `auto` mode or select from 11 Indian languages
- **RAG-grounded responses** — Vertex AI RAG Engine corpus as a native Gemini tool (retrieval)
- **Streaming LLM** — `send_message_stream` with thread + asyncio.Queue pattern (non-blocking)
- **Chat memory** — persistent `ChatSession` keeps full conversation context
- **AudioWorklet** — dedicated audio thread with 20ms PCM chunks, downsampled to 16kHz
- **Latency tracking** — per-response TTFC (Time To First Chunk) and total response time in UI
- **WhatsApp-style UI** — customer bubbles (left), AI bubbles (right) with streaming text

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vanilla JS, AudioWorklet, WebSocket |
| **Server** | FastAPI + Uvicorn, Python 3.12 |
| **STT** | Google Cloud Speech-to-Text V2 (Chirp 3), `us` multi-region |
| **LLM** | Gemini 2.5 Flash via `google-genai` SDK (Vertex AI backend) |
| **RAG** | Vertex AI RAG Engine (corpus as Gemini retrieval tool) |
| **Deploy** | Cloud Build → Cloud Run (`us-central1`) |

## Prerequisites

- Python 3.12+
- Google Cloud SDK (`gcloud auth application-default login`)
- GCP project with:
  - Cloud Speech-to-Text V2 API enabled
  - Vertex AI API enabled
  - Vertex AI RAG Engine corpus provisioned
  - Cloud Run API enabled

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Ashishkamble004/cymbalassist.git
cd cymbalassist
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file (or set environment variables):

```bash
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1
RAG_CORPUS_RESOURCE_NAME=projects/your-project/locations/us-central1/ragCorpora/YOUR_CORPUS_ID
STT_MODEL=chirp_3
STT_LOCATION=us
STT_LANGUAGE=auto
LLM_MODEL=gemini-2.5-flash
```

### 3. Authenticate to GCP

```bash
gcloud auth application-default login
```

### 4. Run locally

```bash
python run.py
```

Open **http://localhost:8000** — click **Connect & Start**, allow mic access, and start speaking.

### 5. Deploy to Cloud Run

```bash
gcloud builds submit --config=cloudbuild.yaml --region=us-central1
```

## Project Structure

```
cymbalassist/
├── app/
│   ├── __init__.py
│   ├── agent.py                # Core pipeline: Streaming STT → RAG → LLM (streaming)
│   ├── config.py               # Settings from env vars / .env
│   ├── server.py               # FastAPI server with WebSocket endpoint
│   └── static/
│       ├── audio-processor.js  # AudioWorklet: 20ms PCM-16 capture at 16kHz
│       └── index.html          # Agent dashboard (WhatsApp-style chat UI)
├── cloudbuild.yaml             # Cloud Build → Cloud Run pipeline
├── Dockerfile                  # Python 3.12-slim container
├── requirements.txt
├── run.py                      # Entry point (uvicorn)
└── README.md
```

## Key Components

| Component | File | Description |
|---|---|---|
| `StreamingSTTManager` | `agent.py` | Manages Chirp 3 streaming via background thread + audio queue |
| `run_agent` | `agent.py` | WebSocket handler: creates genai client, RAG tool, chat session |
| `process_with_rag_llm` | `agent.py` | Streams Gemini response via thread + asyncio.Queue (non-blocking) |
| `PCMProcessor` | `audio-processor.js` | AudioWorklet: accumulates 20ms of audio, downsamples, sends PCM-16 |
| Dashboard | `index.html` | WhatsApp-style chat with streaming bubbles + latency badges |

## WebSocket Protocol

### Client → Server (binary)
Raw PCM-16 audio chunks (16kHz, mono, little-endian).

### Server → Client (JSON)

| `type` | `role` | Description |
|---|---|---|
| `transcription` | `user` | Interim (`is_final: false`) or final (`is_final: true`) transcript |
| `stream_start` | `assistant` | LLM stream beginning — create empty bubble |
| `stream_chunk` | `assistant` | LLM text chunk — append to bubble |
| `stream_end` | `assistant` | LLM stream complete — finalize bubble |
| `error` | — | Error message |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT_ID` | `general-ak` | Google Cloud project |
| `GCP_LOCATION` | `us-central1` | Vertex AI / Cloud Run region |
| `RAG_CORPUS_RESOURCE_NAME` | — | Full resource name of RAG corpus |
| `STT_MODEL` | `chirp_3` | Speech-to-Text model |
| `STT_LOCATION` | `us` | STT multi-region endpoint |
| `STT_LANGUAGE` | `auto` | Language code(s) — `auto` for auto-detect |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port (Cloud Run overrides to 8080) |
