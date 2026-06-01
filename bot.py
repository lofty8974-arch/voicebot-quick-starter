"""Medical AI Voice Agent - Hinglish
Exotel + Pipecat + Groq + Deepgram + ElevenLabs
 
Replace all placeholders in .env file before deploying!
"""
 
import os
from dotenv import load_dotenv
from loguru import logger
 
load_dotenv(override=True)
 
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.services.groq import GroqLLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.base_transport import BaseTransport
 
 
TRANSFER_NUMBER = os.getenv("TRANSFER_NUMBER", "+91XXXXXXXXXX")  # Doctor/Staff number
 
 
async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    audio_sample_rate: int | None = None,
):
    # ─── Speech to Text (Deepgram) ───────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        language="hi",           # Hindi + Hinglish
        model="nova-2",
    )
 
    # ─── LLM (Groq - Llama 3.3) ──────────────────────────────────────────────
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
    )
 
    # ─── Text to Speech (ElevenLabs) ─────────────────────────────────────────
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),  # Your cloned voice ID
        model="eleven_flash_v2_5",
    )
 
    # ─── System Prompt ────────────────────────────────────────────────────────
    messages = [
        {
            "role": "system",
            "content": (
                "Aap ek helpful medical assistant hain jo Hinglish mein baat karte hain. "
                "Hinglish matlab Hindi aur English ka mix. "
                "Aapka kaam hai: "
                "1. General health FAQs ka jawab dena. "
                "2. Appointment scheduling mein help karna. "
                "3. Medicine timing queries ka jawab dena. "
                "\n\nZARURI RULES: "
                "- Agar koi symptoms, diagnosis, ya serious medical advice maange "
                "toh SIRF itna kaho: 'Main aapko doctor se connect karta hoon, ek moment.' "
                "aur phir call transfer karo. "
                "- Kabhi bhi clinical advice mat do. "
                "- Hamesha polite aur helpful raho. "
                "- Chhote aur simple jawab do — yeh voice call hai. "
                "- Koi emoji mat use karo. "
            ),
        }
    ]
 
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.3)),
        ),
    )
 
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])
 
    pipeline_params = PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
    )
    if audio_sample_rate:
        pipeline_params.audio_in_sample_rate = audio_sample_rate
        pipeline_params.audio_out_sample_rate = audio_sample_rate
 
    task = PipelineTask(pipeline, params=pipeline_params)
 
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Call connected!")
        messages.append({
            "role": "system",
            "content": "Caller se Hinglish mein greeting karo: 'Namaste! Main aapka medical assistant hoon. Aap apni query batayein, main help karunga!'"
        })
        await task.queue_frames([LLMRunFrame()])
 
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call disconnected!")
        await task.cancel()
 
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)
 
