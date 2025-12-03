import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from .config import settings

class RateLimitError(Exception):
    pass

def retry_on_rate_limit():
    return retry(
        reraise=True,
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((RateLimitError, requests.exceptions.RequestException)),
    )

@retry_on_rate_limit()
def call_groq(prompt: str, model: str = "llama3-8b-8192", max_tokens: int = 512) -> dict:

    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }

    resp = requests.post(url, json=body, headers=headers, timeout=30)

    if resp.status_code == 429:
        raise RateLimitError("Groq rate limited")
    
    resp.raise_for_status()
    return resp.json()
