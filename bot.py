#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import datetime
import io
import os
import sys
import wave

import aiofiles # Asegúrate de que aiofiles esté instalado si no lo estaba ya
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
# Descomenta la siguiente línea si necesitas el ResamplerProcessor
# from pipecat.processors.audio.resampler import ResamplerProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
# Ya no necesitamos ElevenLabsTTSService ni WhisperSTTService ni OpenAILLMService
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    InputParams,
)
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2) # Asumiendo 16-bit audio
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(websocket_client: WebSocket, stream_sid: str, testing: bool): # call_sid no se usa aquí, lo quité si no es necesario para TwilioFrameSerializer
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False, # Gemini probablemente no quiere encabezados WAV en el stream
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(), # Gemini también tiene VAD, pero esto puede ayudar a controlar el flujo de frames inicial
            vad_audio_passthrough=True, # Asegura que el audio llegue a Gemini
            serializer=TwilioFrameSerializer(stream_sid=stream_sid), # stream_sid es necesario aquí
        ),
    )

    # Inicializar el servicio Gemini Multimodal Live
    # Asegúrate de tener GEMINI_API_KEY en tu .env
    llm_gemini = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        voice_id="Puck",  # Puedes elegir otras voces: Aoede, Charon, Fenrir, Kore
        transcribe_user_audio=True, # Gemini se encargará del STT
        params=InputParams(temperature=0.7)
    )

    # Mensajes del sistema en ESPAÑOL y adaptados para Gemini (usa "user" para el prompt inicial)
    # El prompt que tenías en el ejemplo de Gemini para RAOD System:
    messages = [
        {
            "role": "user", # Gemini usa "user" para el system prompt inicial
            "content": "Eres un agente de IA que pertenece a RAOD System, debes presentarte de tal manera, tienes toda la información referente a la empresa. Habla siempre en español.",
        },
        # O tu prompt anterior adaptado:
        # {
        #     "role": "user",
        #     "content": "Eres Tasha, una asistente de IA muy servicial. Tus respuestas se convertirán a audio, así que no incluyas caracteres especiales. Responde con frases cortas y siempre en español.",
        # },
    ]

    context = OpenAILLMContext(messages)
    # Gemini service crea su propio context aggregator compatible
    context_aggregator = llm_gemini.create_context_aggregator(context)

    audiobuffer = AudioBufferProcessor(user_continuous_stream=not testing)

    pipeline_processors = [
        transport.input(),          # Entrada de audio desde Twilio/Websocket
        context_aggregator.user(),  # Agrega el contexto del usuario (basado en STT interno de Gemini)
        llm_gemini,                 # Procesa con Gemini (STT interno, LLM, TTS interno)
        context_aggregator.assistant(), # Agrega el contexto del asistente (respuesta de Gemini)
    ]

    # IMPORTANTE: Considerar re-muestreo para Twilio si es necesario
    # Gemini produce audio a 24kHz. Twilio usualmente espera 8kHz.
    # Si tienes problemas con el audio en Twilio, descomenta las siguientes líneas
    # e instala el procesador si es necesario (`pip install pipecat-ai[processors]`).
    # from pipecat.processors.audio.resampler import ResamplerProcessor
    # pipeline_processors.append(ResamplerProcessor(target_sample_rate=8000))

    pipeline_processors.extend([
        transport.output(),         # Salida de audio a Twilio/Websocket
        audiobuffer,                # Buffer para grabar (opcional)
    ])

    pipeline = Pipeline(pipeline_processors)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,  # Gemini espera 16kHz
            audio_out_sample_rate=24000, # Gemini produce 24kHz. (Ver nota sobre re-muestreo para Twilio)
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await audiobuffer.start_recording()
        # Mensaje de inicio en ESPAÑOL (Gemini usa 'user' para este tipo de interacciones)
        # messages.append({"role": "user", "content": "Por favor, preséntate al usuario. Recuerda tu nombre y que hablas en español."})
        # O simplemente deja que el system prompt inicial guíe la primera interacción:
        await task.queue_frames([context_aggregator.user().get_context_frame()])


    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        # server_name podría necesitar ser definido de otra manera si websocket_client.client.port no está disponible
        # podrías usar el stream_sid o una parte de él.
        server_name = f"server_twilio_{stream_sid}"
        await save_audio(server_name, audio, sample_rate, num_channels)

    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    await runner.run(task)