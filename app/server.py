"""
FastAPI server for the Call-Center Agent Assist system.

Exposes:
  • GET  /    → serves the agent dashboard (index.html)
  • WS   /ws  → Streaming STT → RAG → LLM pipeline
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(override=True)

from app.agent import run_agent
from app.config import settings

# --------------------------------------------------------------------------- #
#  App bootstrap                                                               #
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Call Center Agent Assist", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  WebSocket endpoint                                                          #
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    stt_model: str = "chirp_3",
    stt_language: str = "auto",
    llm_model: str = "gemini-2.5-flash",
    system_instruction: Optional[str] = None,
):
    await websocket.accept()
    print("WebSocket connection accepted")
    try:
        await run_agent(
            websocket,
            stt_model=stt_model,
            stt_language=stt_language,
            llm_model=llm_model,
            system_instruction=system_instruction,
        )
    except Exception as e:
        print(f"Exception in run_agent: {e}")



# --------------------------------------------------------------------------- #
#  Static files & index                                                        #
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# --------------------------------------------------------------------------- #
#  Entry point (for running directly: python -m app.server)                    #
# --------------------------------------------------------------------------- #
async def main():
    config = uvicorn.Config(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
