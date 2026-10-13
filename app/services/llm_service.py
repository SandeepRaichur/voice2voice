import json
import re
import httpx
from typing import List, Dict, Optional, Tuple, AsyncIterator
from app.config import settings
from app.schemas import ChatMessage, EmotionEstimate, ServiceStatus


class LLMService:
    def __init__(self):
        pass

    @property
    def ollama_base_url(self) -> str:
        return settings.OLLAMA_BASE_URL.rstrip('/')

    @property
    def ollama_model(self) -> str:
        return settings.OLLAMA_MODEL

    @property
    def timeout(self) -> float:
        return settings.OLLAMA_TIMEOUT

    @property
    def system_prompt(self) -> str:
        return settings.SYSTEM_PROMPT

    @property
    def gemini_api_key(self) -> str:
        return settings.get_gemini_api_key()

    @property
    def gemini_model(self) -> str:
        model = settings.GEMINI_MODEL.removeprefix("models/").strip()
        if any(r in model for r in ["2.0-flash", "2.0-flash-lite", "2.0-flash-exp"]):
            return "gemini-3.8-flash"
        return model or "gemini-3.8-flash"

    @property
    def provider(self) -> str:
        return settings.AI_PROVIDER.lower()

    @property
    def active_model(self) -> str:
        if self.provider == "gemini":
            return self.gemini_model
        return self.ollama_model

    @property
    def model(self) -> str:
        return self.active_model

    async def check_health(self) -> ServiceStatus:
        """Check AI brain status based on active provider (Gemini or Ollama)."""
        if self.provider == "gemini":
            key = self.gemini_api_key
            if not key:
                return ServiceStatus(
                    ready=False,
                    details=f"Gemini selected ({self.gemini_model}). API key not configured.",
                    error="Set GEMINI_API_KEY in .env or the in-app Settings modal. (Free key: https://aistudio.google.com/app/apikey)"
                )

            # Validate key with Google Gemini API securely via header
            try:
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}"
                headers = {"x-goog-api-key": key}
                client_timeout = httpx.Timeout(5.0, connect=settings.GEMINI_CONNECT_TIMEOUT, read=5.0)
                async with httpx.AsyncClient(timeout=client_timeout) as client:
                    res = await client.get(test_url, headers=headers)
                    if res.status_code == 200:
                        return ServiceStatus(
                            ready=True,
                            details=f"Google Gemini connected. Model '{self.gemini_model}' (Free Tier) active."
                        )
                    else:
                        try:
                            err_msg = res.json().get("error", {}).get("message", res.text[:120])
                        except Exception:
                            err_msg = res.text[:120]
                        return ServiceStatus(
                            ready=False,
                            details="Google Gemini API key validation failed.",
                            error=f"HTTP {res.status_code}: {err_msg}"
                        )

            except Exception as e:
                return ServiceStatus(
                    ready=False,
                    details="Could not connect to Google Gemini endpoint.",
                    error=str(e)
                )

        # Fallback to local Ollama health check
        return await self._check_ollama_health()

    async def _check_ollama_health(self) -> ServiceStatus:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.ollama_base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    has_model = any(self.ollama_model in m or m.startswith(self.ollama_model.split(":")[0]) for m in models)
                    if has_model:
                        return ServiceStatus(
                            ready=True,
                            details=f"Ollama connected. Model '{self.ollama_model}' is available."
                        )
                    else:
                        available_list = ", ".join(models) if models else "none"
                        return ServiceStatus(
                            ready=False,
                            details=f"Ollama is running, but model '{self.ollama_model}' is not downloaded.",
                            error=f"Run: ollama pull {self.ollama_model} (Available: {available_list})"
                        )
                else:
                    return ServiceStatus(
                        ready=False,
                        details="Ollama HTTP service returned unexpected status.",
                        error=f"HTTP {res.status_code}"
                    )
        except Exception as e:
            return ServiceStatus(
                ready=False,
                details="Ollama is not running at " + self.ollama_base_url,
                error=f"Connection failed: {str(e)}. Start Ollama with 'ollama serve'."
            )

    def _build_emotion_guidance(self, emotion: Optional[EmotionEstimate]) -> str:
        """Generate subtle behavioral guidance based on vocal emotion cues (Section 2 & 5)."""
        if not emotion or not emotion.emotion:
            return ""

        emo = emotion.emotion.lower()
        conf = emotion.confidence

        if emo in ["frustrated", "angry"] and conf > 0.5:
            return (
                "\n[Vocal Emotion Context: Acoustic cues suggest frustration or tension. "
                "Guidance: Remain composed and calm. Acknowledge the difficulty respectfully and focus "
                "on practical solution steps. Do not tell the user how they feel.]"
            )
        elif emo in ["sad", "worried", "anxious"] and conf > 0.5:
            return (
                "\n[Vocal Emotion Context: Acoustic cues suggest sadness or worry. "
                "Guidance: Respond with warmth, patience, and gentle empathy. Keep your pacing calm.]"
            )
        elif emo in ["excited", "happy"] and conf > 0.5:
            return (
                "\n[Vocal Emotion Context: Acoustic cues suggest high energy or excitement. "
                "Guidance: Match their enthusiasm warmly without sounding dramatic or exaggerated.]"
            )
        elif emo in ["confused", "uncertain"] and conf > 0.5:
            return (
                "\n[Vocal Emotion Context: Acoustic cues suggest confusion or questioning intonation. "
                "Guidance: Simplify the explanation. Break the thought down step-by-step.]"
            )
        return ""

    def clean_spoken_response(self, text: str) -> str:
        """Strip markdown symbols, bullets, asterisks, emojis to ensure pure spoken English."""
        if not text:
            return "I didn't quite catch that. Could you please repeat?"

        cleaned = text.strip()
        cleaned = re.sub(r'[*_#`~\[\]]', '', cleaned)
        cleaned = re.sub(r'^\s*[-•*+]\s+', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n+', ' ', cleaned)
        emoji_pattern = re.compile(
            "[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]",
            flags=re.UNICODE
        )
        cleaned = emoji_pattern.sub('', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        return cleaned

    async def generate_response(
        self,
        transcript: str,
        history: List[ChatMessage],
        emotion: Optional[EmotionEstimate] = None
    ) -> Tuple[str, Optional[str], str, str]:
        """Generate response via configured provider (Gemini or Ollama) and return (reply, guidance, provider, model)."""
        if self.provider == "gemini":
            api_key = self.gemini_api_key
            if not api_key:
                if settings.FALLBACK_TO_OLLAMA:
                    ollama_health = await self._check_ollama_health()
                    if ollama_health.ready:
                        reply, guidance = await self._generate_ollama(transcript, history, emotion)
                        return reply, guidance, "ollama", self.ollama_model
                raise RuntimeError(
                    "Gemini provider selected, but GEMINI_API_KEY is not configured. "
                    "Please set your Gemini API key in Settings or .env."
                )

            try:
                reply, guidance = await self._generate_gemini(transcript, history, emotion)
                return reply, guidance, "gemini", self.gemini_model
            except Exception as e:
                if settings.FALLBACK_TO_OLLAMA:
                    ollama_health = await self._check_ollama_health()
                    if ollama_health.ready:
                        reply, guidance = await self._generate_ollama(transcript, history, emotion)
                        return reply, guidance, "ollama", self.ollama_model
                raise RuntimeError(f"Gemini API request failed: {str(e)}")

        elif self.provider == "ollama":
            reply, guidance = await self._generate_ollama(transcript, history, emotion)
            return reply, guidance, "ollama", self.ollama_model

        else:
            raise RuntimeError(f"Unsupported AI provider '{self.provider}'. Supported providers: 'gemini', 'ollama'.")

    async def generate_response_stream(
        self,
        transcript: str,
        history: List[ChatMessage],
        emotion: Optional[EmotionEstimate] = None
    ) -> AsyncIterator[Tuple[str, str, str]]:
        """Yield tokens as they arrive in real-time as (token_text, provider_used, model_used).
        
        If cloud Gemini is selected and fails or times out, seamlessly falls back to Ollama.
        """
        if self.provider == "gemini":
            api_key = self.gemini_api_key
            if not api_key:
                if settings.FALLBACK_TO_OLLAMA:
                    ollama_health = await self._check_ollama_health()
                    if ollama_health.ready:
                        async for token in self._stream_ollama(transcript, history, emotion):
                            yield token, "ollama", self.ollama_model
                        return
                raise RuntimeError(
                    "Gemini provider selected, but GEMINI_API_KEY is not configured. "
                    "Please set your Gemini API key in Settings or .env."
                )

            gemini_streamed_any = False
            try:
                async for token in self._stream_gemini(transcript, history, emotion):
                    gemini_streamed_any = True
                    yield token, "gemini", self.gemini_model
                return
            except Exception as e:
                # If Gemini failed before yielding tokens and fallback is enabled, switch to Ollama
                if not gemini_streamed_any and settings.FALLBACK_TO_OLLAMA:
                    ollama_health = await self._check_ollama_health()
                    if ollama_health.ready:
                        async for token in self._stream_ollama(transcript, history, emotion):
                            yield token, "ollama", self.ollama_model
                        return
                raise RuntimeError(f"Gemini API streaming failed: {str(e)}")

        elif self.provider == "ollama":
            async for token in self._stream_ollama(transcript, history, emotion):
                yield token, "ollama", self.ollama_model

        else:
            raise RuntimeError(f"Unsupported AI provider '{self.provider}'.")

    async def _stream_ollama(
        self,
        transcript: str,
        history: List[ChatMessage],
        emotion: Optional[EmotionEstimate] = None
    ) -> AsyncIterator[str]:
        """Stream tokens incrementally from local Ollama instance."""
        emotion_guidance = self._build_emotion_guidance(emotion)
        system_content = self.system_prompt
        if emotion_guidance:
            system_content += emotion_guidance

        messages_payload = [{"role": "system", "content": system_content}]
        for msg in history:
            messages_payload.append({
                "role": msg.role,
                "content": msg.content
            })
        messages_payload.append({
            "role": "user",
            "content": transcript
        })

        ollama_timeout = httpx.Timeout(
            timeout=30.0,
            connect=4.0,
            read=30.0,
            write=5.0,
            pool=5.0
        )
        max_tokens = getattr(settings, "LLM_MAX_TOKENS", 120)

        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages_payload,
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": max_tokens
                        }
                    }
                ) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        raise RuntimeError(f"Ollama stream error (HTTP {response.status_code}): {err_text.decode('utf-8', errors='ignore')}")
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                chunk = data.get("message", {}).get("content", "")
                                if chunk:
                                    yield chunk
                            except Exception:
                                continue
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_base_url}.")
        except httpx.TimeoutException:
            raise RuntimeError("Ollama streaming timed out.")

    async def _stream_gemini(
        self,
        transcript: str,
        history: List[ChatMessage],
        emotion: Optional[EmotionEstimate] = None
    ) -> AsyncIterator[str]:
        """Stream tokens incrementally from Google Gemini using Server-Sent Events."""
        emotion_guidance = self._build_emotion_guidance(emotion)
        system_content = self.system_prompt
        if emotion_guidance:
            system_content += emotion_guidance

        contents = []
        for msg in history:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": transcript}]
        })

        model = self.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        max_tokens = getattr(settings, "LLM_MAX_TOKENS", 120)
        payload = {
            "systemInstruction": {"parts": [{"text": system_content}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_tokens,
                "topP": 0.95
            }
        }

        client_timeout = httpx.Timeout(
            timeout=25.0,
            connect=settings.GEMINI_CONNECT_TIMEOUT,
            read=12.0,
            write=5.0,
            pool=5.0
        )

        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code == 503:
                        raise RuntimeError("Google Gemini is experiencing high demand (HTTP 503).")
                    elif response.status_code == 429:
                        raise RuntimeError("Google Gemini rate limit exceeded (HTTP 429).")
                    elif response.status_code != 200:
                        err_bytes = await response.aread()
                        raise RuntimeError(f"Gemini API stream returned status {response.status_code}: {err_bytes.decode('utf-8', errors='ignore')[:150]}")

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            if not raw_json:
                                continue
                            try:
                                data = json.loads(raw_json)
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    if parts and "text" in parts[0]:
                                        yield parts[0]["text"]
                            except Exception:
                                continue
        except httpx.ReadTimeout:
            raise RuntimeError(f"Gemini API read timed out.")
        except httpx.ConnectTimeout:
            raise RuntimeError("Gemini API connection timed out.")
        except httpx.ConnectError:
            raise RuntimeError("Could not connect to Google Gemini.")

    async def _generate_gemini(
        self,
        transcript: str,
        history: List[ChatMessage],
        emotion: Optional[EmotionEstimate] = None
    ) -> Tuple[str, Optional[str]]:
        """Query Google Gemini API endpoint securely using header authentication with explicit timeouts."""
        emotion_guidance = self._build_emotion_guidance(emotion)
        system_content = self.system_prompt
        if emotion_guidance:
            system_content += emotion_guidance

        contents = []
        for msg in history:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": transcript}]
        })

        model = self.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_content}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 300,
                "topP": 0.95
            }
        }

        client_timeout = httpx.Timeout(
            timeout=settings.GEMINI_READ_TIMEOUT + 5.0,
            connect=settings.GEMINI_CONNECT_TIMEOUT,
            read=settings.GEMINI_READ_TIMEOUT,
            write=settings.GEMINI_WRITE_TIMEOUT,
            pool=5.0
        )

        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                res = await client.post(url, headers=headers, json=payload)
        except httpx.ReadTimeout:
            raise RuntimeError(f"Gemini API read timed out after {settings.GEMINI_READ_TIMEOUT}s.")
        except httpx.ConnectTimeout:
            raise RuntimeError(f"Gemini API connection timed out after {settings.GEMINI_CONNECT_TIMEOUT}s.")
        except httpx.TimeoutException:
            raise RuntimeError("Gemini API request timed out.")
        except httpx.ConnectError:
            raise RuntimeError("Could not establish connection to Google Gemini API.")

        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if not candidates:
                prompt_feedback = data.get("promptFeedback", {})
                block_reason = prompt_feedback.get("blockReason", "Safety policy")
                raise RuntimeError(f"Gemini generation blocked: {block_reason}")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts or "text" not in parts[0]:
                raise RuntimeError("Gemini returned empty content parts.")
            raw_reply = parts[0]["text"]
            clean_reply = self.clean_spoken_response(raw_reply)
            return clean_reply, emotion_guidance
        elif res.status_code == 503:
            raise RuntimeError("Google Gemini model is currently experiencing high demand (HTTP 503).")
        elif res.status_code == 429:
            raise RuntimeError("Google Gemini rate limit exceeded (HTTP 429).")
        elif res.status_code == 404:
            raise RuntimeError(f"Gemini model '{model}' is not available (HTTP 404).")
        else:
            try:
                err_json = res.json()
                err_msg = err_json.get("error", {}).get("message", res.text[:200])
            except Exception:
                err_msg = res.text[:200]
            raise RuntimeError(f"Gemini API returned status {res.status_code}: {err_msg}")

    async def _generate_ollama(
        self,
        transcript: str,
        history: List[ChatMessage],
        emotion: Optional[EmotionEstimate] = None
    ) -> Tuple[str, Optional[str]]:
        """Query local Ollama instance with granular timeouts."""
        emotion_guidance = self._build_emotion_guidance(emotion)
        system_content = self.system_prompt
        if emotion_guidance:
            system_content += emotion_guidance

        messages_payload = [{"role": "system", "content": system_content}]
        for msg in history:
            messages_payload.append({
                "role": msg.role,
                "content": msg.content
            })
        messages_payload.append({
            "role": "user",
            "content": transcript
        })

        ollama_timeout = httpx.Timeout(
            timeout=self.timeout,
            connect=5.0,
            read=self.timeout,
            write=10.0,
            pool=5.0
        )
        try:
            async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                res = await client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages_payload,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 250
                        }
                    }
                )

                if res.status_code == 200:
                    data = res.json()
                    raw_reply = data.get("message", {}).get("content", "")
                    clean_reply = self.clean_spoken_response(raw_reply)
                    return clean_reply, emotion_guidance
                else:
                    raise RuntimeError(f"Ollama returned status {res.status_code}: {res.text}")

        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_base_url}. Ensure Ollama is running ('ollama serve') "
                f"and that model '{self.ollama_model}' is downloaded."
            )
        except httpx.TimeoutException:
            raise RuntimeError(f"Ollama request timed out after {self.timeout}s.")


llm_service = LLMService()
