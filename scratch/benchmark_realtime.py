"""Empirical Benchmark for Real-Time Low-Latency Voice Turn."""

import asyncio
import json
import time
import httpx
import websockets

HTTP_URL = "http://127.0.0.1:3000"
WS_URL = "ws://127.0.0.1:3000/ws/conversation"


async def run_benchmark():
    print("=" * 60)
    print("REAL-TIME LOW-LATENCY VOICE PIPELINE BENCHMARK")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # 1. Health check
        t0 = time.perf_counter()
        res = await client.get(f"{HTTP_URL}/api/status")
        t_health = (time.perf_counter() - t0) * 1000
        print(f"[1] Status Check ({t_health:.1f}ms): {res.json()['server']}")

        # 2. Switch to Ollama
        print("[2] Setting Provider to Ollama Local Brain...")
        await client.post(f"{HTTP_URL}/api/settings", json={"provider": "ollama"})
        res_settings = await client.get(f"{HTTP_URL}/api/settings")
        cfg = res_settings.json()
        print(f"    Active Settings: Provider={cfg['provider']} | Model={cfg['ollama_model']} | TTS={cfg['tts_engine']} | Voice={cfg.get('edge_voice')}")

        # 3. WebSocket Real-Time Turn
        print("\n[3] Connecting to Full-Duplex Real-Time WebSocket...")
        async with websockets.connect(WS_URL) as ws:
            ready_raw = await ws.recv()
            ready_msg = json.loads(ready_raw)
            print(f"    Handshake received: {ready_msg['type']} (TTS: {ready_msg['config']['tts_engine']})")

            print("\n--- Test: Real-Time Stream Turn (Ollama qwen2.5:3b) ---")
            t_turn_start = time.perf_counter()
            await ws.send(json.dumps({
                "type": "text_turn",
                "text": "Say 'Good morning! Ready to help.' in four words."
            }))

            first_token_time = None
            first_audio_time = None
            first_sentence = ""
            total_audio_bytes = 0
            token_count = 0

            while True:
                msg_raw = await ws.recv()
                msg = json.loads(msg_raw)
                mtype = msg.get("type")

                if mtype == "llm_token":
                    if first_token_time is None:
                        first_token_time = (time.perf_counter() - t_turn_start) * 1000
                    token_count += 1

                elif mtype == "sentence_start":
                    if not first_sentence:
                        first_sentence = msg.get("text", "")

                elif mtype == "audio_chunk":
                    if first_audio_time is None:
                        first_audio_time = (time.perf_counter() - t_turn_start) * 1000
                    total_audio_bytes += len(msg.get("data", ""))

                elif mtype == "turn_complete":
                    t_turn_end = (time.perf_counter() - t_turn_start) * 1000
                    m = msg.get("metrics", {})
                    print(f"    * Provider Used:               {msg.get('provider')} ({msg.get('model')})")
                    print(f"    * Time to First Token (TTFT):  {first_token_time:.1f} ms")
                    print(f"    * Time to First Audio (TTFA):   {first_audio_time:.1f} ms")
                    print(f"    * Total Turn Duration:          {t_turn_end:.1f} ms")
                    print(f"    * First Sentence Delivered:     \"{first_sentence}\"")
                    print(f"    * Total Tokens Received:        {token_count}")
                    print(f"    * Total Synthesized Audio:      {total_audio_bytes} base64 chars")
                    print(f"    * Server Telemetry Metrics:     {m}")
                    break

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
