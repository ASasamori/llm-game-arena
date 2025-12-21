import os
import json
import httpx
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Log API key status on module load
if OPENROUTER_API_KEY:
    logger.info(f"OpenRouter API Key loaded: {OPENROUTER_API_KEY[:10]}...")
if GEMINI_API_KEY:
    logger.info(f"Gemini API Key loaded: {GEMINI_API_KEY[:10]}...")

async def get_llm_response(
    model_id: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> Optional[str]:
    """
    Send a request to either OpenRouter or Gemini API based on model_id.
    """
    if model_id.startswith("gemini-") or model_id.startswith("gemma-"):
        return await get_gemini_response(model_id, system_prompt, user_message, temperature, max_tokens)
    else:
        return await get_openrouter_response(model_id, system_prompt, user_message, temperature, max_tokens)

class RateLimitError(Exception):
    """Raised when the LLM API rate limit is exceeded."""
    def __init__(self, message="Rate limit exceeded", retry_after=5):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

async def get_gemini_response(
    model_id: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> Optional[str]:
    """Send a request to Google Gemini API."""
    logger.info(f"🚀 Gemini Request to: {model_id}")
    
    
    url = f"{GEMINI_API_URL}/{model_id}:generateContent?key={GEMINI_API_KEY}"
    
    # MOCK TEST: Force rate limit once
    # if not hasattr(get_gemini_response, "has_failed"):
    #     get_gemini_response.has_failed = True
    #     logger.warning("🧪 TESTING: Simulating Rate Limit 429")
    #     raise RateLimitError("Simulated Rate Limit", retry_after=5)

    # Combined prompt for Gemini
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"SYSTEM INSTRUCTION: {system_prompt}\n\nUSER MESSAGE: {user_message}"}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 429:
                logger.warning("⏳ Gemini Rate Limit Hit (429)")
                raise RateLimitError("Gemini Rate Limit Exceeded", retry_after=10)

            if response.status_code != 200:
                logger.error(f"❌ Gemini HTTP {response.status_code}: {response.text[:500]}")
                return None
                
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                
                # Estimate tokens (approx 4 chars per token)
                input_tokens = len(user_message) //4
                output_tokens = len(content) //4
                total_tokens = input_tokens + output_tokens
                
                logger.info(f"🔢 Assuming 4 char/token: Input {input_tokens} + Output {output_tokens} = Total {total_tokens} tokens")
                logger.info(f"✅ Got Gemini response ({len(content)} chars)")
                return content
            else:
                logger.warning(f"⚠️ No candidates in Gemini response: {data}")
                return None
    except RateLimitError:
        raise  # Re-raise to be handled by engine/main
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
        return None

async def get_openrouter_response(
    model_id: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> Optional[str]:
    """Send a request to OpenRouter API."""
    logger.info(f"🚀 OpenRouter Request to: {model_id}")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Mafia AI Simulation"
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"❌ OpenRouter HTTP {response.status_code}: {response.text[:500]}")
                return None
                
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                logger.info(f"✅ Got OpenRouter response ({len(content)} chars)")
                return content
            else:
                logger.warning(f"⚠️ No choices in OpenRouter response: {data}")
                return None
    except Exception as e:
        logger.error(f"❌ OpenRouter error: {e}")
        return None


def parse_ai_response(response_text: str) -> dict:
    """
    Parse the AI's response to extract inner_monologue and public_speech.
    
    The AI is instructed to respond in JSON format:
    {
        "inner_monologue": "...",
        "public_speech": "..."
    }

    Falls back to treating the entire response as public speech if parsing fails.
    """
    logger.debug(f"Parsing AI response ({len(response_text) if response_text else 0} chars)")
    
    if not response_text:
        logger.warning("Empty response text to parse")
        return {
            "inner_monologue": "(No response)",
            "public_speech": "(No response)"
        }

    # Try to find JSON in the response
    try:
        # Look for JSON block
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            parsed = json.loads(json_str)

            result = {
                "inner_monologue": parsed.get("inner_monologue", "(Thinking...)"),
                "public_speech": parsed.get("public_speech", response_text)
            }
            logger.debug(f"✅ Parsed JSON successfully")
            return result
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ JSON parse failed: {e}")

    # Fallback: treat entire response as public speech
    logger.info("Using fallback: entire response as public speech")
    return {
        "inner_monologue": "(Thinking...)",
        "public_speech": response_text.strip()
    }
