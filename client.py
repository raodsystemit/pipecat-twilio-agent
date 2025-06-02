#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import datetime
import io
import os
import sys
import wave
import xml.etree.ElementTree as ET
from uuid import uuid4

import aiofiles
import aiohttp
from dotenv import load_dotenv
from loguru import logger

# --- PIPECAT IMPORTS ---
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, TransportMessageUrgentFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
# from pipecat.processors.audio.resampler import ResamplerProcessor # Asegúrate de que esté importado
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    InputParams,
)
from pipecat.transports.network.websocket_client import (
    WebsocketClientParams,
    WebsocketClientTransport,
)

load_dotenv(override=True)

# Configuración del logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG") # Nivel DEBUG para ver todos los logs


DEFAULT_CLIENT_DURATION = 30


async def download_twiml(server_url: str, client_name: str) -> str:
    logger.debug(f"[{client_name}] Attempting to download TwiML from: {server_url}")
    async with aiohttp.ClientSession() as session:
        async with session.post(server_url) as response:
            logger.debug(f"[{client_name}] TwiML request status: {response.status}")
            response.raise_for_status()
            twiml = await response.text()
            logger.debug(f"[{client_name}] Received TwiML (first 100 chars): {twiml[:100]}...")
            return twiml


def get_stream_url_from_twiml(twiml: str, client_name: str) -> str:
    logger.debug(f"[{client_name}] Parsing TwiML to find stream URL...")
    root = ET.fromstring(twiml)
    stream_element = root.find(".//Stream")
    if stream_element is None or stream_element.get("url") is None:
        error_msg = f"Could not find Stream URL in TwiML: {twiml}"
        logger.error(f"[{client_name}] {error_msg}")
        raise ValueError(error_msg)
    url = stream_element.get("url")
    logger.info(f"[{client_name}] Extracted WebSocket Stream URL: {url}")
    return url


async def save_audio(client_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{client_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"[{client_name}] Merged audio saved to {filename}")
    else:
        logger.debug(f"[{client_name}] No audio data to save (empty audio frame).")


async def run_client(client_name: str, server_url: str, duration_secs: int):
    logger.info(f"[{client_name}] Starting client to connect to {server_url} for {duration_secs}s")
    try:
        twiml = await download_twiml(server_url, client_name)
        stream_url = get_stream_url_from_twiml(twiml, client_name)
    except Exception as e:
        logger.error(f"[{client_name}] Failed to get TwiML or stream URL: {e}. Aborting client.")
        return

    stream_sid = str(uuid4())
    logger.debug(f"[{client_name}] Generated Stream SID: {stream_sid}")

    logger.debug(f"[{client_name}] Initializing WebsocketClientTransport to {stream_url}")
    transport = WebsocketClientTransport(
        uri=stream_url,
        params=WebsocketClientParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=TwilioFrameSerializer(stream_sid),
            # Parámetros VAD actualizados (quitar vad_enabled y vad_audio_passthrough si tu versión de Pipecat lo soporta)
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.5)),
        ),
    )
    logger.debug(f"[{client_name}] WebsocketClientTransport initialized.")

    # --- SELECCIONA LA PIPELINE A PROBAR ---

    # --- OPCIÓN 1: PIPELINE SIMPLIFICADA (para depurar AttributeError) ---
    USE_SIMPLIFIED_PIPELINE = True # CAMBIA A False PARA USAR LA PIPELINE COMPLETA
    # --- FIN OPCIÓN 1 ---

    if USE_SIMPLIFIED_PIPELINE:
        logger.info(f"[{client_name}] Using SIMPLIFIED pipeline.")
        pipeline_processors = [
            transport.input(),
            transport.output(),
        ]
        active_pipeline_params = PipelineParams(
            audio_in_sample_rate=8000,  # TODO: Ajusta a lo que el bot realmente envía (Twilio suele ser 8k)
            audio_out_sample_rate=8000, # Debe coincidir con _in si no hay procesamiento intermedio
            allow_interruptions=True,
        )
    else:
        logger.info(f"[{client_name}] Using FULL Gemini pipeline.")
        # Gemini Multimodal Live para el CLIENTE
        llm_client_gemini = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            voice_id="Puck",
            transcribe_user_audio=True,
            params=InputParams(temperature=0.9)
        )
        messages = [{
            "role": "user",
            "content": "Eres un niño o niña de 8 años llamado Alex. Eres curioso y un poco travieso. Un profesor o profesora te explicará conceptos nuevos. Haz preguntas como lo haría un niño de tu edad. Habla siempre en español.",
        }]
        context = OpenAILLMContext(messages)
        context_aggregator = llm_client_gemini.create_context_aggregator(context)
        audiobuffer = AudioBufferProcessor(user_continuous_stream=False)

        pipeline_processors = [
            transport.input(),
            # TODO: Considera un ResamplerProcessor aquí si el audio del bot (servidor)
            # no está a 16kHz, que es lo que Gemini prefiere en su entrada.
            # ResamplerProcessor(target_sample_rate=16000),
            transport.output(),
            audiobuffer,
        ]
        active_pipeline_params = PipelineParams(
            # TODO: Ajusta audio_in_sample_rate a lo que el bot realmente envía.
            # Si añades un Resampler a 16k después del input, esta sería la tasa original del bot.
            audio_in_sample_rate=16000,
            audio_out_sample_rate=8000, # Después del Resampler de salida a 8kHz
            allow_interruptions=True,
        )

    logger.debug(f"[{client_name}] Initializing Pipeline with {len(pipeline_processors)} processors...")
    pipeline = Pipeline(pipeline_processors)
    logger.debug(f"[{client_name}] Pipeline initialized.")

    logger.debug(f"[{client_name}] Initializing PipelineTask with params: {active_pipeline_params}")
    task = PipelineTask(pipeline, params=active_pipeline_params)
    logger.debug(f"[{client_name}] PipelineTask initialized: {task.id if task else 'None'}")

    # Event Handlers
    @transport.event_handler("on_connected")
    async def on_connected(transport_instance, client_websocket):
        logger.info(f"[{client_name}] Event: Websocket connected! Session: {client_websocket}")
        if not USE_SIMPLIFIED_PIPELINE and 'audiobuffer' in locals():
            await audiobuffer.start_recording()
            logger.debug(f"[{client_name}] AudioBuffer recording started (full pipeline).")

        logger.debug(f"[{client_name}] Sending Twilio 'connected' message.")
        connected_message = TransportMessageUrgentFrame(
            message={"event": "connected", "protocol": "Call", "version": "1.0.0"}
        )
        await transport_instance.output().send_message(connected_message)

        logger.debug(f"[{client_name}] Sending Twilio 'start' message with Stream SID: {stream_sid}")
        start_message = TransportMessageUrgentFrame(
            message={"event": "start", "streamSid": stream_sid, "start": {"streamSid": stream_sid}}
        )
        await transport_instance.output().send_message(start_message)
        logger.info(f"[{client_name}] Twilio start messages sent.")

        if not USE_SIMPLIFIED_PIPELINE and 'context_aggregator' in locals():
            logger.debug(f"[{client_name}] Kicking off client's conversation context (full pipeline).")
            # Esto es para que el LLM del cliente genere su primera frase si así está diseñado el prompt.
            # await task.queue_frames([context_aggregator.user().get_context_frame()]) # Podría necesitar un StartFrame o un UserStartedSpeakingFrame
            # Considera enviar un frame vacío o un evento específico para iniciar el LLM del cliente si es necesario
            pass


    @transport.event_handler("on_error")
    async def on_error(transport_instance, error_message):
        logger.error(f"[{client_name}] Event: Websocket error! Message: {error_message}")

    @transport.event_handler("on_disconnected")
    async def on_disconnected(transport_instance, reason=None):
        logger.warning(f"[{client_name}] Event: Websocket disconnected! Reason: {reason}")

    if not USE_SIMPLIFIED_PIPELINE and 'audiobuffer' in locals():
        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            # Este audio es el que el cliente *envía* (su propia voz generada por Gemini y re-muestreada)
            await save_audio(f"{client_name}_output", audio, sample_rate, num_channels)

    async def end_call_after_duration():
        logger.debug(f"[{client_name}] Call will end in {duration_secs} seconds.")
        await asyncio.sleep(duration_secs)
        logger.info(f"[{client_name}] Duration ended. Sending EndFrame to task: {task.id if task else 'None'}")
        if task and task.is_running: # Solo envía EndFrame si la tarea existe y está corriendo
            await task.queue_frame(EndFrame())
        else:
            logger.warning(f"[{client_name}] Task not running or not initialized, cannot send EndFrame.")


    runner = PipelineRunner()
    logger.info(f"[{client_name}] Starting PipelineRunner for task: {task.id if task else 'None'}")
    try:
        # Ejecutar la pipeline y la finalización de la llamada concurrentemente
        await asyncio.gather(runner.run(task), end_call_after_duration())
    except Exception as e:
        logger.error(f"[{client_name}] Critical error during pipeline execution or end_call: {e}", exc_info=True)
        # Asegúrate de que la tarea se cancele si hay un error aquí
        if task and task.is_running:
            await task.cancel()
            logger.info(f"[{client_name}] Task {task.id} cancelled due to critical error.")
    finally:
        logger.info(f"[{client_name}] Client run finished.")


async def main():
    parser = argparse.ArgumentParser(description="Pipecat Twilio Chatbot Client con Logs")
    parser.add_argument("-u", "--url", type=str, required=True, help="specify the server URL")
    parser.add_argument("-c", "--clients", type=int, default=1, help="number of concurrent clients")
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=DEFAULT_CLIENT_DURATION,
        help=f"duration of each client in seconds (default: {DEFAULT_CLIENT_DURATION})",
    )
    args = parser.parse_args() # Usar parse_args() si no esperas argumentos desconocidos

    logger.info(f"Starting main function, preparing to launch {args.clients} client(s).")

    client_tasks = []
    for i in range(args.clients):
        # Crear un nombre de cliente único para cada instancia
        unique_client_name = f"client_{i}_{str(uuid4())[:8]}" # Más único que solo el índice
        client_tasks.append(
            asyncio.create_task(run_client(unique_client_name, args.url, args.duration))
        )

    try:
        await asyncio.gather(*client_tasks)
    except Exception as e:
        logger.error(f"Error gathering client tasks: {e}", exc_info=True)
    logger.info("All client tasks gathered.")


if __name__ == "__main__":
    asyncio.run(main())