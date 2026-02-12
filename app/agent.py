"""
Agent for call-center agent assist.

Pipeline:  Audio In → Streaming STT → RAG Context Injection → LLM → Text Out
(No TTS — the agent reads Gemini responses on the dashboard.)

Uses native WebSocket with raw PCM audio (16kHz, 16-bit, mono).
Implements streaming Speech-to-Text for low-latency transcription.
"""

import asyncio
import json
import queue
import threading
from typing import Optional

from loguru import logger
from fastapi import WebSocket, WebSocketDisconnect

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from google import genai
from google.genai import types

from app.config import settings


# --------------------------------------------------------------------------- #
#  System prompt for the call-center agent-assist LLM                         #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a call-center agent assistant. The customer is on the phone.
Give the shortest possible answer the agent can read aloud (1-3 sentences max).
Use the RAG knowledge base when available. No bullet points, no lists, no markdown.
"""


# --------------------------------------------------------------------------- #
#  Streaming STT Manager - handles continuous audio streaming                  #
# --------------------------------------------------------------------------- #
class StreamingSTTManager:
    """Manages streaming Speech-to-Text with Chirp 3 model.
    
    Based on: https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3#perform_streaming_speech_recognition
    """
    
    def __init__(
        self,
        project_id: str,
        location: str,
        model: str,
        language_code: str,
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
        self.language_code = language_code
        
        # Parse language codes (handle comma-separated string)
        if "," in language_code:
            self.language_codes_list = [lang.strip() for lang in language_code.split(",")]
        else:
            self.language_codes_list = [language_code]
        
        # Audio queue for streaming
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._transcript_callback = None
        self._stream_thread = None
        
        # Create client with regional endpoint (us-central1 for Chirp 3)
        self.client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{location}-speech.googleapis.com"
            )
        )
        
        # Recognizer path for Chirp 3 streaming
        self.recognizer = f"projects/{project_id}/locations/{location}/recognizers/_"
        
        # Streaming config using explicit decoding (required for raw PCM from browser)
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=cloud_speech.RecognitionConfig(
                # Use explicit decoding for raw PCM audio (16-bit linear, 16kHz)
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    audio_channel_count=1,
                ),
                # Language codes - Chirp 3 supports multiple languages
                language_codes=self.language_codes_list,
                # Model - chirp_3 for Chirp 3
                model=model,
            ),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                # Enable interim results for low-latency streaming
                interim_results=True,
            ),
        )
    
    def _request_generator(self):
        """Generate streaming requests from audio queue."""
        # First request must contain the config
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer,
            streaming_config=self.streaming_config,
        )
        
        # Subsequent requests contain audio data
        while self._is_running:
            try:
                audio_chunk = self._audio_queue.get(timeout=0.01)
                if audio_chunk is None:  # Poison pill
                    break
                yield cloud_speech.StreamingRecognizeRequest(audio=audio_chunk)
            except queue.Empty:
                continue
    
    def _stream_worker(self, loop: asyncio.AbstractEventLoop):
        """Worker thread for streaming recognition."""
        while self._is_running:
            try:
                responses = self.client.streaming_recognize(
                    requests=self._request_generator()
                )
                
                for response in responses:
                    if not self._is_running:
                        break
                        
                    for result in response.results:
                        if result.alternatives:
                            transcript = result.alternatives[0].transcript
                            is_final = result.is_final
                            
                            if self._transcript_callback and transcript:
                                # Schedule callback in the async event loop
                                asyncio.run_coroutine_threadsafe(
                                    self._transcript_callback(transcript, is_final),
                                    loop
                                )
                
                # If stream ends normally, we might want to break or restart depending on logic.
                # Usually streaming_recognize returns when stream closes.
                # If we are still running, it implies a disconnect, so we loop to reconnect.
                if not self._is_running:
                    break

            except Exception as e:
                logger.error(f"Streaming STT error: {e}")
                # Prevent tight loop on permanent errors
                if self._is_running:
                    logger.info("Restarting STT stream in 1 second...")
                    import time
                    time.sleep(1)
                else:
                    break
    
    def start(self, transcript_callback, loop: asyncio.AbstractEventLoop):
        """Start the streaming recognition."""
        self._transcript_callback = transcript_callback
        self._is_running = True
        self._stream_thread = threading.Thread(
            target=self._stream_worker, 
            args=(loop,),
            daemon=True
        )
        self._stream_thread.start()
        logger.info("Streaming STT started")
    
    def add_audio(self, audio_data: bytes):
        """Add audio data to the stream."""
        if self._is_running:
            self._audio_queue.put(audio_data)
    
    def stop(self):
        """Stop the streaming recognition."""
        self._is_running = False
        self._audio_queue.put(None)  # Poison pill
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
        logger.info("Streaming STT stopped")


# --------------------------------------------------------------------------- #
#  Agent runner - handles WebSocket, Streaming STT, RAG, and LLM              #
# --------------------------------------------------------------------------- #
async def run_agent(
    websocket: WebSocket,
    stt_model: str = "chirp_3",
    stt_language: str = "en-IN",
    llm_model: str = "gemini-2.5-flash",
    system_instruction: Optional[str] = None,
):
    """Handle WebSocket connection with Streaming STT → RAG → LLM pipeline."""
    
    system_prompt = system_instruction or SYSTEM_PROMPT
    stt_location = settings.STT_LOCATION
    
    # Initialize google.genai client with Vertex AI backend
    client = genai.Client(
        vertexai=True,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
    )
    
    # RAG tool — Gemini will automatically retrieve from the corpus
    rag_tool = types.Tool(
        retrieval=types.Retrieval(
            vertex_rag_store=types.VertexRagStore(
                rag_resources=[
                    types.VertexRagStoreRagResource(
                        rag_corpus=settings.RAG_CORPUS_RESOURCE_NAME,
                    )
                ],
            )
        )
    )
    
    # Create persistent chat session with RAG grounding
    chat = client.chats.create(
        model=llm_model,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[rag_tool],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            ],
        ),
    )
    
    logger.info(f"Agent started: STT={stt_model}@{stt_location}, LLM={llm_model}, language={stt_language}")
    
    # Track transcript state
    current_transcript = ""
    last_final_transcript = ""
    pending_llm_task = None
    
    async def handle_transcript(transcript: str, is_final: bool):
        """Handle incoming transcripts from streaming STT."""
        nonlocal current_transcript, last_final_transcript, pending_llm_task
        
        current_transcript = transcript
        
        # Send interim transcripts to client for real-time display
        await websocket.send_text(json.dumps({
            "type": "transcription",
            "role": "user",
            "text": transcript,
            "is_final": is_final,
        }))
        
        if is_final and transcript.strip():
            last_final_transcript = transcript
            logger.info(f"Final transcript: {transcript}")
            
            # Cancel any pending LLM task
            if pending_llm_task and not pending_llm_task.done():
                pending_llm_task.cancel()
            
            # Process with RAG and LLM
            pending_llm_task = asyncio.create_task(
                process_with_rag_llm(websocket, chat, transcript)
            )
    
    # Create streaming STT manager
    stt_manager = StreamingSTTManager(
        project_id=settings.GCP_PROJECT_ID,
        location=stt_location,
        model=stt_model,
        language_code=stt_language,
    )
    
    try:
        # Start streaming STT
        loop = asyncio.get_event_loop()
        stt_manager.start(handle_transcript, loop)
        
        while True:
            try:
                # Receive audio data from WebSocket
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                
                # Feed audio to streaming STT
                stt_manager.add_audio(data)
                
            except asyncio.TimeoutError:
                # No data received, connection might be idle
                continue
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Agent error: {e}")
    finally:
        stt_manager.stop()
        if pending_llm_task and not pending_llm_task.done():
            pending_llm_task.cancel()


async def process_with_rag_llm(
    websocket: WebSocket,
    chat,
    transcript: str,
):
    """Stream transcript through Gemini with RAG grounding, pushing chunks over WebSocket."""
    try:
        loop = asyncio.get_event_loop()
        chunk_q: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()

        def _stream_in_thread():
            """Run the blocking gRPC streaming iterator in a background thread."""
            try:
                for chunk in chat.send_message_stream(transcript):
                    if cancelled.is_set():
                        break
                    text = chunk.text or ""
                    if text:
                        asyncio.run_coroutine_threadsafe(chunk_q.put(text), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(chunk_q.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(chunk_q.put(None), loop)  # sentinel

        # Signal stream start
        await websocket.send_text(json.dumps({"type": "stream_start", "role": "assistant"}))

        thread = threading.Thread(target=_stream_in_thread, daemon=True)
        thread.start()

        full_parts: list[str] = []
        while True:
            item = await chunk_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            full_parts.append(item)
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "role": "assistant",
                "text": item,
            }))

        await websocket.send_text(json.dumps({"type": "stream_end", "role": "assistant"}))

        full_text = "".join(full_parts).strip()
        if full_text:
            logger.info(f"LLM response (streamed): {full_text[:120]}...")
        else:
            logger.warning(f"LLM returned empty stream for: {transcript[:60]}")

    except asyncio.CancelledError:
        cancelled.set()
        logger.debug("LLM task cancelled (new transcript received)")
    except Exception as e:
        logger.error(f"LLM error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": f"LLM error: {str(e)}",
            }))
        except:
            pass
