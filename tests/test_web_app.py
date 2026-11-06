import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import yaml

import web_app


class PublicWebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = web_app.Settings(
            openai_api_key="test-key",
            app_hmac_key=b"x" * 32,
            monthly_budget_usd=1,
            input_price_per_million=0.15,
            output_price_per_million=0.60,
            captcha_site_key="site-key",
            captcha_secret_key="secret-key",
            database_path=Path(self.temp_dir.name) / "usage.sqlite3",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_request_reservations_enforce_monthly_budget(self):
        _, reservation = web_app.reserve_request(
            self.settings, "ip-hash", "test prompt"
        )
        limited_settings = replace(
            self.settings, monthly_budget_usd=reservation * 1.5
        )

        with self.assertRaises(HTTPException) as raised:
            web_app.reserve_request(limited_settings, "ip-hash", "test prompt")

        self.assertEqual(raised.exception.status_code, 429)

    def test_request_reservations_enforce_per_ip_rate_limit(self):
        with patch.object(web_app, "MAX_REQUESTS_PER_MINUTE", 1):
            web_app.reserve_request(self.settings, "ip-hash", "first")
            with self.assertRaises(HTTPException) as raised:
                web_app.reserve_request(self.settings, "ip-hash", "second")

        self.assertEqual(raised.exception.status_code, 429)

    def test_prompt_length_is_limited(self):
        with self.assertRaises(ValueError):
            web_app.ChatRequest(prompt="x" * 2001, captcha_token="captcha")

    def test_chat_rejects_oversized_http_body(self):
        with TestClient(web_app.app) as client:
            response = client.post(
                "/api/chat",
                content=b"x" * (web_app.MAX_REQUEST_BODY_BYTES + 1),
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 413)

    def test_example_environment_file_uses_credential_placeholders(self):
        example = (web_app.ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertRegex(
            example, r"(?m)^OPENAI_API_KEY=replace_with_server_secret\r?$"
        )
        self.assertNotRegex(
            example, r"(?m)^OPENAI_API_KEY=sk-"
        )
        for key in (
            "APP_HMAC_KEY",
            "CAPTCHA_SITE_KEY",
            "CAPTCHA_SECRET_KEY",
            "OPENAI_MONTHLY_BUDGET_USD",
            "OPENAI_INPUT_PRICE_PER_MILLION",
            "OPENAI_OUTPUT_PRICE_PER_MILLION",
        ):
            self.assertRegex(example, rf"(?m)^{key}=")

    def test_configuration_diagnostics_name_missing_settings_without_values(self):
        names = (
            "OPENAI_API_KEY",
            "APP_HMAC_KEY",
            "CAPTCHA_SITE_KEY",
            "CAPTCHA_SECRET_KEY",
            "OPENAI_MONTHLY_BUDGET_USD",
            "OPENAI_INPUT_PRICE_PER_MILLION",
            "OPENAI_OUTPUT_PRICE_PER_MILLION",
        )
        with patch.dict("os.environ", {name: "" for name in names}):
            missing, invalid = web_app.configuration_issues()

        self.assertEqual(missing, list(names))
        self.assertEqual(invalid, [])

    def test_example_placeholders_are_not_treated_as_real_credentials(self):
        placeholders = {
            "OPENAI_API_KEY": "replace_with_server_secret",
            "APP_HMAC_KEY": "replace_with_server_secret",
            "CAPTCHA_SITE_KEY": "replace_with_turnstile_site_key",
            "CAPTCHA_SECRET_KEY": "replace_with_turnstile_secret_key",
            "OPENAI_MONTHLY_BUDGET_USD": "",
            "OPENAI_INPUT_PRICE_PER_MILLION": "",
            "OPENAI_OUTPUT_PRICE_PER_MILLION": "",
        }
        with patch.dict("os.environ", placeholders):
            missing, _ = web_app.configuration_issues()

        self.assertIn("OPENAI_API_KEY", missing)
        self.assertIn("CAPTCHA_SITE_KEY", missing)

    def test_public_page_is_available_without_model_secrets(self):
        with TestClient(web_app.app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI Data Agent", response.text)
        self.assertIn('id="chat-form"', response.text)
        self.assertIn('id="message-list"', response.text)
        self.assertIn('id="submit"', response.text)
        self.assertIn("ETL workflow", response.text)

    def test_chat_uses_text_only_model_request(self):
        create_response = Mock(
            return_value=SimpleNamespace(
                output_text="Here is an explanation.",
                usage=SimpleNamespace(input_tokens=20, output_tokens=8),
            )
        )
        provider_client = SimpleNamespace(
            responses=SimpleNamespace(create=create_response),
            close=lambda: None,
        )
        environment = {
            "OPENAI_API_KEY": "test-key",
            "APP_HMAC_KEY": "x" * 32,
            "CAPTCHA_SITE_KEY": "site-key",
            "CAPTCHA_SECRET_KEY": "secret-key",
            "OPENAI_MONTHLY_BUDGET_USD": "1",
            "OPENAI_INPUT_PRICE_PER_MILLION": "0.15",
            "OPENAI_OUTPUT_PRICE_PER_MILLION": "0.60",
            "DATA_AGENT_DB_PATH": str(Path(self.temp_dir.name) / "chat.sqlite3"),
        }
        with patch.dict("os.environ", environment), patch.object(
            web_app, "verify_captcha"
        ) as verify_captcha, patch.object(
            web_app, "OpenAI", return_value=provider_client
        ):
            with TestClient(web_app.app) as client:
                response = client.post(
                    "/api/chat",
                    json={"prompt": "Explain a CSV", "captcha_token": "valid"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Here is an explanation.")
        self.assertEqual(create_response.call_args.kwargs["model"], web_app.MODEL)
        self.assertNotIn("tools", create_response.call_args.kwargs)
        self.assertEqual(
            verify_captcha.call_args.args[3], "testserver"
        )

    def test_api_ask_preserves_structured_response_contract(self):
        create_response = Mock(
            return_value=SimpleNamespace(
                output_text="Use a staging table to validate data.",
                usage=SimpleNamespace(input_tokens=20, output_tokens=8),
            )
        )
        provider_client = SimpleNamespace(
            responses=SimpleNamespace(create=create_response),
            close=lambda: None,
        )
        environment = {
            "OPENAI_API_KEY": "test-key",
            "APP_HMAC_KEY": "x" * 32,
            "CAPTCHA_SITE_KEY": "site-key",
            "CAPTCHA_SECRET_KEY": "secret-key",
            "OPENAI_MONTHLY_BUDGET_USD": "1",
            "OPENAI_INPUT_PRICE_PER_MILLION": "0.15",
            "OPENAI_OUTPUT_PRICE_PER_MILLION": "0.60",
            "DATA_AGENT_DB_PATH": str(Path(self.temp_dir.name) / "ask.sqlite3"),
        }
        with patch.dict("os.environ", environment), patch.object(
            web_app, "verify_captcha"
        ), patch.object(web_app, "OpenAI", return_value=provider_client):
            with TestClient(web_app.app) as client:
                response = client.post(
                    "/api/ask",
                    json={
                        "prompt": "How do I validate ETL data?",
                        "captcha_token": "valid",
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["prompt"], "How do I validate ETL data?")
        self.assertEqual(body["reply"], "Use a staging table to validate data.")
        self.assertTrue(body["timestamp"])

    def test_api_ask_validates_prompt_length(self):
        with TestClient(web_app.app) as client:
            response = client.post(
                "/api/ask",
                json={
                    "prompt": "x" * 2001,
                    "captcha_token": "valid",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"error": "Prompt is too long (max 2000 characters)"},
        )

    def test_api_ask_returns_structured_error_for_empty_prompt(self):
        with TestClient(web_app.app) as client:
            response = client.post(
                "/api/ask",
                json={"prompt": " ", "captcha_token": "valid"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Prompt is required"})

    def test_api_ask_requires_captcha_token(self):
        with TestClient(web_app.app) as client:
            response = client.post(
                "/api/ask",
                json={"prompt": "Explain a SQL join"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"error": "A valid CAPTCHA token is required"},
        )

    def test_chat_fails_closed_when_secrets_are_missing(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "",
                "APP_HMAC_KEY": "",
                "CAPTCHA_SITE_KEY": "",
                "CAPTCHA_SECRET_KEY": "",
                "OPENAI_MONTHLY_BUDGET_USD": "",
                "OPENAI_INPUT_PRICE_PER_MILLION": "",
                "OPENAI_OUTPUT_PRICE_PER_MILLION": "",
            },
        ):
            with TestClient(web_app.app) as client:
                response = client.post(
                    "/api/chat",
                    json={"prompt": "hello", "captcha_token": "unused"},
                )

        self.assertEqual(response.status_code, 503)

    def test_public_configuration_does_not_claim_missing_service_is_ready(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "",
                "APP_HMAC_KEY": "",
                "CAPTCHA_SITE_KEY": "",
                "CAPTCHA_SECRET_KEY": "",
                "OPENAI_MONTHLY_BUDGET_USD": "",
                "OPENAI_INPUT_PRICE_PER_MILLION": "",
                "OPENAI_OUTPUT_PRICE_PER_MILLION": "",
            },
        ):
            with TestClient(web_app.app) as client:
                response = client.get("/api/config")

        self.assertFalse(response.json()["ready"])

    def test_public_configuration_exposes_only_captcha_site_key(self):
        environment = {
            "OPENAI_API_KEY": "server-only-openai-key",
            "APP_HMAC_KEY": "x" * 32,
            "CAPTCHA_SITE_KEY": "public-site-key",
            "CAPTCHA_SECRET_KEY": "server-only-captcha-key",
            "OPENAI_MONTHLY_BUDGET_USD": "2",
            "OPENAI_INPUT_PRICE_PER_MILLION": "0.15",
            "OPENAI_OUTPUT_PRICE_PER_MILLION": "0.60",
        }
        with patch.dict("os.environ", environment):
            with TestClient(web_app.app) as client:
                response = client.get("/api/config")

        self.assertTrue(response.json()["ready"])
        self.assertEqual(response.json()["captchaSiteKey"], "public-site-key")
        serialized = response.text
        self.assertNotIn("server-only-openai-key", serialized)
        self.assertNotIn("server-only-captcha-key", serialized)
        self.assertNotIn("x" * 32, serialized)

    def test_render_blueprint_keeps_secrets_external_and_usage_persistent(self):
        blueprint = yaml.safe_load(
            (web_app.ROOT / "render.yaml").read_text(encoding="utf-8")
        )
        service = blueprint["services"][0]
        environment = {entry["key"]: entry for entry in service["envVars"]}

        self.assertEqual(service["healthCheckPath"], "/health")
        self.assertEqual(service["disk"]["mountPath"], "/opt/render/project/src/data")
        self.assertIn("--forwarded-allow-ips=\"*\"", service["startCommand"])
        for key in (
            "OPENAI_API_KEY",
            "APP_HMAC_KEY",
            "CAPTCHA_SITE_KEY",
            "CAPTCHA_SECRET_KEY",
            "OPENAI_MONTHLY_BUDGET_USD",
            "OPENAI_INPUT_PRICE_PER_MILLION",
            "OPENAI_OUTPUT_PRICE_PER_MILLION",
        ):
            self.assertTrue(environment[key]["sync"] is False, key)


if __name__ == "__main__":
    unittest.main()
