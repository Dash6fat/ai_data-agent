import hashlib
import hmac
import json
import logging
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from openai import APIError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
load_dotenv(ROOT / ".env")
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
MAX_PROMPT_CHARS = 2000
MAX_REQUEST_BODY_BYTES = 16_384
MAX_OUTPUT_TOKENS = 400
MODEL = "gpt-4o-mini"
MAX_REQUESTS_PER_MINUTE = 5
MAX_REQUESTS_PER_DAY = 30
SYSTEM_PROMPT = (
    "You are a helpful assistant for the AI Data Agent project. Explain data "
    "analysis and ETL concepts and answer questions clearly. You do not have "
    "access to databases, files, tools, or external URLs, and must not claim "
    "to have performed actions."
)
logger = logging.getLogger("ai_data_agent.public")

app = FastAPI(title="AI Data Agent", docs_url=None, redoc_url=None)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: FastAPIRequest, exception: RequestValidationError
):
    if request.url.path != "/api/ask":
        return await request_validation_exception_handler(request, exception)

    errors = exception.errors()
    prompt_errors = [
        error for error in errors if error.get("loc", ())[-1:] == ("prompt",)
    ]
    captcha_errors = [
        error
        for error in errors
        if error.get("loc", ())[-1:] == ("captcha_token",)
    ]
    if prompt_errors:
        if any(error.get("type") == "string_too_long" for error in prompt_errors):
            message = "Prompt is too long (max 2000 characters)"
        elif any(error.get("type") == "string_too_short" for error in prompt_errors):
            message = "Prompt is required"
        elif any(
            "Prompt is required" in error.get("msg", "")
            for error in prompt_errors
        ):
            message = "Prompt is required"
        else:
            message = "Prompt must be text"
    elif captcha_errors:
        message = "A valid CAPTCHA token is required"
    else:
        message = "Invalid request"
    return JSONResponse(status_code=400, content={"error": message})


class RequestSizeLimitMiddleware:
    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("path") != "/api/chat"
            or scope.get("method") != "POST"
        ):
            await self.application(scope, receive, send)
            return

        content_length = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            if not content_length.isdigit():
                await self._reject_invalid_request(send)
                return
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                await self._reject_large_request(send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_REQUEST_BODY_BYTES:
                await self._reject_large_request(send)
                return
            if not message.get("more_body", False):
                break

        body_bytes = bytes(body)
        replayed = False

        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {
                    "type": "http.request",
                    "body": body_bytes,
                    "more_body": False,
                }
            return await receive()

        await self.application(scope, replay_receive, send)

    @staticmethod
    async def _reject_large_request(send):
        body = b'{"detail":"Request body is too large."}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _reject_invalid_request(send):
        body = b'{"detail":"Invalid Content-Length header."}'
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


app.add_middleware(RequestSizeLimitMiddleware)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    app_hmac_key: bytes
    monthly_budget_usd: float
    input_price_per_million: float
    output_price_per_million: float
    captcha_site_key: str
    captcha_secret_key: str
    database_path: Path


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    captcha_token: str = Field(min_length=1, max_length=4096)

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt is required")
        return value


def _configured_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.lower().startswith("replace_with_"):
        return ""
    return value


def _positive_float(name: str) -> float:
    value = os.environ.get(name, "")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=503, detail=f"Server setting {name} is not configured."
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise HTTPException(
            status_code=503, detail=f"Server setting {name} is invalid."
        )
    return parsed


def get_settings() -> Settings:
    required = (
        "OPENAI_API_KEY",
        "APP_HMAC_KEY",
        "CAPTCHA_SITE_KEY",
        "CAPTCHA_SECRET_KEY",
    )
    missing = [name for name in required if not _configured_value(name)]
    if missing:
        raise HTTPException(
            status_code=503,
            detail="The public chat service is not configured yet.",
        )

    hmac_key = _configured_value("APP_HMAC_KEY").encode("utf-8")
    if len(hmac_key) < 32:
        raise HTTPException(
            status_code=503, detail="Server setting APP_HMAC_KEY is too short."
        )

    database_path = Path(
        os.environ.get(
            "DATA_AGENT_DB_PATH", str(ROOT / "data" / "web_usage.sqlite3")
        )
    )
    return Settings(
        openai_api_key=_configured_value("OPENAI_API_KEY"),
        app_hmac_key=hmac_key,
        monthly_budget_usd=_positive_float("OPENAI_MONTHLY_BUDGET_USD"),
        input_price_per_million=_positive_float(
            "OPENAI_INPUT_PRICE_PER_MILLION"
        ),
        output_price_per_million=_positive_float(
            "OPENAI_OUTPUT_PRICE_PER_MILLION"
        ),
        captcha_site_key=_configured_value("CAPTCHA_SITE_KEY"),
        captcha_secret_key=_configured_value("CAPTCHA_SECRET_KEY"),
        database_path=database_path,
    )


def configuration_issues() -> tuple[list[str], list[str]]:
    required = (
        "OPENAI_API_KEY",
        "APP_HMAC_KEY",
        "CAPTCHA_SITE_KEY",
        "CAPTCHA_SECRET_KEY",
        "OPENAI_MONTHLY_BUDGET_USD",
        "OPENAI_INPUT_PRICE_PER_MILLION",
        "OPENAI_OUTPUT_PRICE_PER_MILLION",
    )
    missing = [name for name in required if not _configured_value(name)]
    invalid = []

    hmac_key = _configured_value("APP_HMAC_KEY")
    if hmac_key and len(hmac_key) < 32:
        invalid.append("APP_HMAC_KEY (must be at least 32 characters)")

    for name in (
        "OPENAI_MONTHLY_BUDGET_USD",
        "OPENAI_INPUT_PRICE_PER_MILLION",
        "OPENAI_OUTPUT_PRICE_PER_MILLION",
    ):
        value = _configured_value(name)
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            invalid.append(name + " (must be a positive number)")
            continue
        if not math.isfinite(parsed) or parsed <= 0:
            invalid.append(name + " (must be a positive number)")

    return missing, invalid


def _connect(settings: Settings) -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS public_requests (
            request_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            day_key TEXT NOT NULL,
            month_key TEXT NOT NULL,
            ip_hash TEXT NOT NULL,
            prompt_chars INTEGER NOT NULL,
            reserved_usd REAL NOT NULL,
            actual_usd REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS public_requests_ip_created "
        "ON public_requests(ip_hash, created_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS public_requests_month "
        "ON public_requests(month_key, status)"
    )
    return connection


def _hash_ip(settings: Settings, ip_address: str) -> str:
    return hmac.new(
        settings.app_hmac_key, ip_address.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def reserve_request(
    settings: Settings, ip_hash: str, prompt: str
) -> tuple[str, float]:
    now = time.time()
    now_utc = datetime.fromtimestamp(now, tz=timezone.utc)
    month_key = now_utc.strftime("%Y-%m")
    day_key = now_utc.strftime("%Y-%m-%d")
    request_id = str(uuid.uuid4())
    encoding = tiktoken.get_encoding("o200k_base")
    estimated_input_tokens = (
        len(encoding.encode(SYSTEM_PROMPT))
        + len(encoding.encode(prompt))
        + 64
    )
    estimated_cost = (
        estimated_input_tokens * settings.input_price_per_million
        + MAX_OUTPUT_TOKENS * settings.output_price_per_million
    ) / 1_000_000

    connection = _connect(settings)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM public_requests WHERE created_at < ?",
                (now - 35 * 24 * 60 * 60,),
            )
            minute_count = connection.execute(
                "SELECT COUNT(*) FROM public_requests "
                "WHERE ip_hash = ? AND created_at >= ?",
                (ip_hash, now - 60),
            ).fetchone()[0]
            day_count = connection.execute(
                "SELECT COUNT(*) FROM public_requests "
                "WHERE ip_hash = ? AND day_key = ?",
                (ip_hash, day_key),
            ).fetchone()[0]
            month_reserved = connection.execute(
                "SELECT COALESCE(SUM(reserved_usd), 0) FROM public_requests "
                "WHERE month_key = ?",
                (month_key,),
            ).fetchone()[0]

            if minute_count >= MAX_REQUESTS_PER_MINUTE:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please try again in a minute.",
                )
            if day_count >= MAX_REQUESTS_PER_DAY:
                raise HTTPException(
                    status_code=429,
                    detail="Daily request limit reached. Please try again tomorrow.",
                )
            if month_reserved + estimated_cost > settings.monthly_budget_usd:
                raise HTTPException(
                    status_code=429,
                    detail="The service has reached its monthly usage limit.",
                )

            connection.execute(
                """
                INSERT INTO public_requests
                    (request_id, created_at, day_key, month_key, ip_hash,
                     prompt_chars, reserved_usd, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'reserved')
                """,
                (
                    request_id,
                    now,
                    day_key,
                    month_key,
                    ip_hash,
                    len(prompt),
                    estimated_cost,
                ),
            )
        return request_id, estimated_cost
    finally:
        connection.close()


def finish_request(
    settings: Settings, request_id: str, actual_cost: float, status: str
) -> None:
    connection = _connect(settings)
    try:
        connection.execute(
            "UPDATE public_requests SET actual_usd = ?, reserved_usd = ?, "
            "status = ? WHERE request_id = ?",
            (
                actual_cost,
                actual_cost,
                status,
                request_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def verify_captcha(
    settings: Settings,
    token: str,
    ip_address: str,
    expected_hostname: str,
) -> None:
    payload = urlencode(
        {
            "secret": settings.captcha_secret_key,
            "response": token,
            "remoteip": ip_address,
        }
    ).encode("ascii")
    request = Request(
        TURNSTILE_VERIFY_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
    except (URLError, TimeoutError, UnicodeError, json.JSONDecodeError) as exc:
        logger.error("captcha_verification_unavailable error=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="CAPTCHA verification is unavailable."
        ) from exc

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=503, detail="CAPTCHA verification returned an invalid response."
        )
    if result.get("success") is not True:
        raise HTTPException(status_code=403, detail="CAPTCHA verification failed.")
    if str(result.get("hostname", "")).lower() != expected_hostname.lower():
        raise HTTPException(status_code=403, detail="CAPTCHA hostname did not match.")


@app.middleware("http")
async def security_headers(request: FastAPIRequest, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' "
        "https://challenges.cloudflare.com; frame-src "
        "https://challenges.cloudflare.com; style-src 'self'; "
        "connect-src 'self' https://challenges.cloudflare.com; "
        "object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/app.js", include_in_schema=False)
def javascript():
    return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")


@app.get("/style.css", include_in_schema=False)
def stylesheet():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@app.get("/api/config")
def public_config():
    site_key = _configured_value("CAPTCHA_SITE_KEY") or None
    missing, invalid = configuration_issues()
    return {
        "captchaSiteKey": site_key,
        "ready": not missing and not invalid,
        "missingSettings": missing,
        "invalidSettings": invalid,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(payload: ChatRequest, request: FastAPIRequest):
    return {"answer": _ask_model(payload, request)}


@app.post("/api/ask")
def ask(payload: ChatRequest, request: FastAPIRequest):
    reply = _ask_model(payload, request)
    return {
        "prompt": payload.prompt,
        "reply": reply,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _ask_model(payload: ChatRequest, request: FastAPIRequest) -> str:
    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"
    ip_hash = _hash_ip(settings, client_ip)
    try:
        verify_captcha(
            settings, payload.captcha_token, client_ip, request.url.hostname or ""
        )
    except HTTPException as exc:
        logger.warning(
            "captcha_verification_failed ip_hash=%s status=%d",
            ip_hash,
            exc.status_code,
        )
        raise
    request_id, reserved_cost = reserve_request(settings, ip_hash, payload.prompt)
    settled = False

    try:
        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=20,
            max_retries=0,
        )
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=payload.prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        finally:
            client.close()

        if response.usage is None:
            raise HTTPException(
                status_code=502, detail="The model did not return usage data."
            )
        actual_cost = (
            response.usage.input_tokens * settings.input_price_per_million
            + response.usage.output_tokens * settings.output_price_per_million
        ) / 1_000_000
        finish_request(settings, request_id, actual_cost, "complete")
        settled = True
        logger.info(
            "public_chat_complete request_id=%s ip_hash=%s prompt_chars=%d "
            "input_tokens=%d output_tokens=%d cost_usd=%.8f",
            request_id,
            ip_hash,
            len(payload.prompt),
            response.usage.input_tokens,
            response.usage.output_tokens,
            actual_cost,
        )
        return response.output_text
    except APIError as exc:
        logger.error(
            "public_chat_provider_error request_id=%s error=%s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="The model request failed. Please try again."
        ) from exc
    finally:
        if not settled:
            finish_request(settings, request_id, reserved_cost, "failed")
