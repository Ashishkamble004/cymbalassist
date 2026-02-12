# Call Center Agent Assist — STT → RAG → Gemini (Pipecat)

Real-time call-center agent-assist system built with [Pipecat](https://github.com/pipecat-ai/pipecat).
Customer speech is transcribed live, grounded via Vertex AI RAG Engine, and answered by Gemini —
all displayed on the agent's dashboard.

## Architecture

```
Customer Audio (mic) ──► FastAPIWebsocketTransport
                              │
                         SileroVAD (turn detection)
                              │
                         GoogleSTTService (chirp_3)
                              │
                         RAGProcessor (Vertex AI RAG Engine)
                              │
                         GoogleLLMService (Gemini Flash)
                              │
                         TranscriptProcessor
                              │
                    Agent Dashboard (WebSocket)
                    ┌─────────┴─────────┐
                    │  Customer         │  AI Response
                    │  Transcript       │  (RAG-grounded)
                    └───────────────────┘
```

**Pipeline** (no TTS — agent reads responses on screen):

```
transport.input() → STT → RAGProcessor → transcript.user()
    → context_aggregator.user() → LLM → transcript.assistant()
    → context_aggregator.assistant() → transport.output()
```

## Prerequisites

- Python 3.10+
- Google Cloud SDK (`gcloud auth application-default login`)
- GCP project with:
  - Gemini API key
  - Vertex AI RAG Engine corpus provisioned
  - Cloud Speech-to-Text API enabled

## Quick Start

### 1. Clone & install

```bash
cd streaming-stt-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your keys:
#   GEMINI_API_KEY, GCP_PROJECT_ID, RAG_CORPUS_RESOURCE_NAME
```

### 3. Authenticate to GCP

```bash
gcloud auth application-default login
```

### 4. Run

```bash
python run.py
```

Open **http://localhost:8000** — enter your Gemini API key and click **Connect & Start**.

## Project Structure

```
streaming-stt-rag/
├── app/
│   ├── agent.py              # Pipecat pipeline (STT → RAG → LLM)
│   ├── config.py             # Settings from .env
│   ├── server.py             # FastAPI server (/ws, /connect, /)
│   ├── services/
│   │   └── rag_service.py    # Vertex AI RAG Engine client
│   └── static/
│       └── index.html        # Agent dashboard (RTVI client)
├── .env.example
├── requirements.txt
├── run.py
└── README.md
```

## Key Components

| Component | Description |
|---|---|
| `FastAPIWebsocketTransport` | Pipecat transport — handles audio I/O via WebSocket |
| `GoogleSTTService` | Google Cloud Speech-to-Text (chirp_3 model) |
| `RAGProcessor` | Custom `FrameProcessor` — runs Vertex AI RAG retrieval per turn |
| `GoogleLLMService` | Gemini Flash via Pipecat — generates RAG-grounded responses |
| `TranscriptProcessor` | Captures user & assistant text, sends to dashboard |
| `SileroVADAnalyzer` | Voice Activity Detection for turn management |

## Reference

Built following the patterns from:
[gemini_live_stt_pipecat](https://github.com/Ashishkamble004/gemini_live_stt_pipecat)
(STT + LLM pipeline, adapted for call-center RAG use case without TTS)
# cymbalassist
