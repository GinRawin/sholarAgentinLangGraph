from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from json import JSONDecodeError
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def masked(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def main() -> int:
    load_dotenv()

    provider = os.getenv("SHOLAR_LLM_PROVIDER", "")
    model = os.getenv("SHOLAR_LLM_MODEL", "")
    api_key = os.getenv("SHOLAR_API_KEY", "")
    base_url = os.getenv("SHOLAR_BASE_URL", "")
    timeout = float(os.getenv("SHOLAR_TEST_TIMEOUT", "30"))

    print(f"provider={provider}")
    print(f"model={model}")
    print(f"base_url={base_url}")
    print(f"api_key={masked(api_key)}")
    print(f"timeout={timeout}s")

    if provider == "placeholder":
        print("placeholder provider does not require a network test")
        return 0
    if not api_key:
        print("missing SHOLAR_API_KEY")
        return 1
    if not base_url:
        print("missing SHOLAR_BASE_URL")
        return 1

    body = {
        "model": model or "deepseek-chat",
        "instructions": "You are a connectivity test assistant.",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Reply with a short JSON object.",
                    }
                ],
            }
        ],
        "text": {"format": {"type": "text"}},
        "store": False,
        "temperature": 0.2,
    }

    request = urllib.request.Request(
        url=base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.perf_counter() - started_at
            raw = response.read()
            text = raw.decode("utf-8", errors="replace")
            print(f"status={response.status}")
            print(f"elapsed={elapsed:.2f}s")
            print(f"content_type={response.headers.get('Content-Type', '')}")
            print(f"body_preview={text[:1000]}")
            try:
                payload = json.loads(text)
            except JSONDecodeError:
                print("json_parse=failed")
                return 2
            print("json_parse=ok")
            print(f"top_level_keys={list(payload) if isinstance(payload, dict) else type(payload).__name__}")
            return 0
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started_at
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"http_error={exc.code}")
        print(f"elapsed={elapsed:.2f}s")
        print(f"content_type={exc.headers.get('Content-Type', '')}")
        print(f"body_preview={detail[:1000]}")
        return 3
    except urllib.error.URLError as exc:
        elapsed = time.perf_counter() - started_at
        print(f"url_error={exc.reason}")
        print(f"elapsed={elapsed:.2f}s")
        return 4
    except TimeoutError:
        elapsed = time.perf_counter() - started_at
        print("timeout_error=true")
        print(f"elapsed={elapsed:.2f}s")
        return 5


if __name__ == "__main__":
    sys.exit(main())
