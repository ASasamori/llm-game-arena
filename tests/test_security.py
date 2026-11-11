"""Security boundaries exercised without credentials, provider calls, or services."""

from __future__ import annotations

from html.parser import HTMLParser
from http.client import HTTPMessage
import io
import json
import os
import re
import unittest
from unittest.mock import MagicMock, patch
from urllib.request import Request
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("ARENA_API_KEYS", "local-dev-key")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from game_arena.openspiel import jev
from orchestrator import api_gateway
from orchestrator.models import Base, Run, RunConfiguration


class _ScriptNonces(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.nonces: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self.nonces.append(dict(attrs).get("nonce"))


class ApiBoundarySecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.key = "security-test-key-not-a-provider-credential"
        self.headers = {"Authorization": "Bearer " + self.key}
        session_patch = patch.object(
            api_gateway, "get_session", side_effect=self.sessions
        )
        self.session_factory = session_patch.start()
        self.addCleanup(session_patch.stop)
        key_patch = patch.object(api_gateway, "API_KEYS", {self.key})
        key_patch.start()
        self.addCleanup(key_patch.stop)
        redis_patch = patch.object(
            api_gateway.redis_conn, "eval", return_value=[1, 1, 1]
        )
        self.redis_eval = redis_patch.start()
        self.addCleanup(redis_patch.stop)
        self.client = TestClient(api_gateway.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def create_run(self) -> str:
        run_id = str(uuid.uuid4())
        with self.sessions() as session:
            session.add(
                Run(run_id=run_id, run_type="PVP", num_episodes=1, status="PENDING")
            )
            session.flush()
            session.add(
                RunConfiguration(
                    run_id=run_id,
                    payload={
                        "game": "tic_tac_toe",
                        "model_names": ["MOCK:a", "MOCK:b"],
                        "api_key": "CONFIGURATION_CREDENTIAL_SENTINEL",  # pragma: allowlist secret -- synthetic test credential or sentinel only
                    },
                )
            )
            session.commit()
        return run_id

    def test_every_versioned_api_route_rejects_missing_credentials_before_io(self):
        tested: set[tuple[str, str]] = set()
        http_methods = {
            "get",
            "put",
            "post",
            "delete",
            "options",
            "head",
            "patch",
            "trace",
        }
        # OpenAPI includes operations registered through lazily included routers;
        # app.routes alone does not contain their expanded APIRoute instances.
        for template, operations in api_gateway.app.openapi()["paths"].items():
            if not template.startswith("/v1/"):
                continue
            path = re.sub(r"\{[^{}]+\}", lambda _: str(uuid.uuid4()), template)
            for operation in sorted(set(operations) & http_methods):
                method = operation.upper()
                tested.add((method, template))
                with self.subTest(method=method, path=template):
                    response = self.client.request(method, path, json={})
                    self.assertEqual(response.status_code, 401, response.text)
                    self.assertNotIn(self.key, response.text)
        self.assertIn(("POST", "/v1/runs"), tested)
        self.assertIn(("GET", "/v1/runs/{run_id}/report"), tested)
        self.assertIn(("GET", "/v1/runs/{run_id}/export"), tested)
        self.assertIn(("GET", "/v1/episodes/{episode_id}/trace"), tested)
        self.session_factory.assert_not_called()
        self.redis_eval.assert_not_called()

    def test_query_strings_and_cookies_cannot_authenticate(self):
        run_id = str(uuid.uuid4())
        for path in (
            "/v1/games",
            f"/v1/runs/{run_id}/report",
            f"/v1/runs/{run_id}/export",
        ):
            with self.subTest(path=path):
                response = self.client.get(
                    path,
                    params={"api_key": self.key, "key": self.key, "token": self.key},
                    headers={"Cookie": "api_key=" + self.key},
                )
                self.assertEqual(response.status_code, 401)
                self.assertNotIn(self.key, response.text)
        self.session_factory.assert_not_called()
        self.redis_eval.assert_not_called()

    def test_malformed_credentials_never_fall_back_to_another_identity(self):
        headers = (
            {"Authorization": "Basic " + self.key},
            {"Authorization": "Bearer "},
            {"Authorization": "Bearer " + self.key + "-wrong"},
            {**self.headers, "x-api-key": "wrong-key"},
        )
        for credentials in headers:
            with self.subTest(headers=credentials):
                response = self.client.get(
                    "/v1/observability/metrics", headers=credentials
                )
                self.assertEqual(response.status_code, 401)
        self.session_factory.assert_not_called()

    def test_only_explicit_header_credentials_authenticate(self):
        for credentials in (self.headers, {"x-api-key": self.key}):
            with self.subTest(headers=credentials):
                response = self.client.get(
                    "/v1/observability/metrics", headers=credentials
                )
                self.assertEqual(response.status_code, 200)
                self.assertNotIn(self.key, response.text)

    def test_empty_key_configuration_does_not_enable_the_documented_dev_key(self):
        with patch.object(api_gateway, "API_KEYS", set()):
            for credentials in (
                {"Authorization": "Bearer local-dev-key"},
                {"x-api-key": "local-dev-key"},
            ):
                with self.subTest(headers=credentials):
                    response = self.client.get(
                        "/v1/observability/metrics", headers=credentials
                    )
                    self.assertEqual(response.status_code, 401)
        self.session_factory.assert_not_called()

    def test_exhausted_rate_limit_blocks_work_without_putting_key_in_redis(self):
        self.redis_eval.return_value = [0, 60, 30000]
        response = self.client.get("/v1/games", headers=self.headers)
        self.assertEqual(response.status_code, 429)
        self.session_factory.assert_not_called()
        self.redis_eval.assert_called_once()
        self.assertNotIn(self.key, str(self.redis_eval.call_args))

    def test_report_and_terminal_content_are_private_and_not_html(self):
        run_id = self.create_run()
        for format_name, content_type in (
            ("json", "application/json"),
            ("text", "text/plain"),
        ):
            with self.subTest(format=format_name):
                response = self.client.get(
                    f"/v1/runs/{run_id}/report",
                    params={"format": format_name},
                    headers=self.headers,
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(
                    response.headers["content-type"].startswith(content_type)
                )
                self.assertEqual(response.headers["cache-control"], "no-store")
                self.assertEqual(response.headers["referrer-policy"], "no-referrer")
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                self.assertNotIn("CONFIGURATION_CREDENTIAL_SENTINEL", response.text)
                self.assertNotIn(self.key, response.text)

    def test_viewer_csp_uses_fresh_nonces_and_disallows_external_execution(self):
        nonces = []
        for _ in range(2):
            response = self.client.get("/reports")
            self.assertEqual(response.status_code, 200)
            policy = response.headers["content-security-policy"]
            self.assertIn("default-src 'none'", policy)
            self.assertIn("connect-src 'self'", policy)
            self.assertIn("frame-ancestors 'none'", policy)
            self.assertIn("base-uri 'none'", policy)
            self.assertIn("form-action 'none'", policy)
            script_policy = next(
                item.strip() for item in policy.split(";") if "script-src" in item
            )
            self.assertNotIn("'unsafe-inline'", script_policy)
            self.assertNotIn("'unsafe-eval'", script_policy)
            matches = re.findall(r"'nonce-([^']+)'", script_policy)
            self.assertEqual(len(matches), 1)
            parser = _ScriptNonces()
            parser.feed(response.text)
            self.assertTrue(parser.nonces)
            self.assertEqual(set(parser.nonces), {matches[0]})
            nonces.append(matches[0])
        self.assertNotEqual(nonces[0], nonces[1])
        self.session_factory.assert_not_called()

    def test_unexpected_database_failure_does_not_disclose_credentials(self):
        secret = "DATABASE_PASSWORD_MUST_NOT_REACH_THE_CLIENT"  # pragma: allowlist secret -- synthetic test credential or sentinel only
        self.session_factory.side_effect = RuntimeError(
            f"database postgresql://arena:{secret}@internal-db/arena failed"
        )
        with self.assertLogs(api_gateway.LOGGER, level="INFO") as captured:
            response = self.client.get(
                f"/v1/runs/{uuid.uuid4()}/report", headers=self.headers
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["status_code"], 500)
        self.assertNotIn(secret, response.text)
        self.assertNotIn("internal-db", response.text)
        self.assertNotIn("postgresql://", response.text)
        logs = json.dumps([record.__dict__ for record in captured.records], default=str)
        self.assertNotIn(secret, logs)
        self.assertNotIn("internal-db", logs)
        self.assertNotIn("postgresql://", logs)

    def test_queue_failure_keeps_recovery_id_without_disclosing_connection_secret(self):
        secret = "QUEUE_PASSWORD_MUST_NOT_REACH_CLIENT_OR_LOGS"  # pragma: allowlist secret -- synthetic test credential or sentinel only
        with (
            patch.object(
                api_gateway,
                "_enqueue_with_backoff",
                side_effect=RuntimeError(f"redis://:{secret}@private-queue failed"),
            ),
            self.assertLogs(api_gateway.LOGGER, level="INFO") as captured,
        ):
            response = self.client.post(
                "/v1/runs",
                headers=self.headers,
                json={"env": "tic_tac_toe", "model_names": ["MOCK:a", "MOCK:b"]},
            )
        self.assertEqual(response.status_code, 503)
        recovery = response.json()["error"]
        self.assertEqual(str(uuid.UUID(recovery["run_id"])), recovery["run_id"])
        self.assertEqual(recovery["status_url"], "/v1/runs/" + recovery["run_id"])
        self.assertNotIn(secret, response.text)
        self.assertNotIn("private-queue", response.text)
        logs = json.dumps([record.__dict__ for record in captured.records], default=str)
        self.assertNotIn(secret, logs)
        self.assertNotIn("private-queue", logs)

    def test_unexpected_failure_is_contained_before_asgi_server_logging(self):
        secret = "UPSTREAM_PASSWORD_MUST_NOT_ESCAPE_TO_ASGI_SERVER"  # pragma: allowlist secret -- synthetic test credential or sentinel only
        self.session_factory.side_effect = RuntimeError(
            f"database connection failed with password {secret}"
        )
        # The default client propagates any exception escaping the ASGI app,
        # just as an outer server logger would receive the original exception.
        strict_client = TestClient(api_gateway.app)
        self.addCleanup(strict_client.close)
        request_id = str(uuid.uuid4())
        with self.assertLogs(api_gateway.LOGGER, level="INFO") as captured:
            response = strict_client.get(
                f"/v1/runs/{uuid.uuid4()}/report",
                headers={**self.headers, "x-request-id": request_id},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["code"], "db_error")
        self.assertEqual(response.json()["request_id"], request_id)
        self.assertEqual(response.headers["x-request-id"], request_id)
        self.assertEqual(
            response.headers["x-api-gateway"], api_gateway.API_GATEWAY_NAME
        )
        self.assertNotIn(secret, response.text)
        logs = json.dumps([record.__dict__ for record in captured.records], default=str)
        self.assertNotIn(secret, logs)

    def test_worker_trigger_failure_keeps_run_without_disclosing_provider_secret(self):
        secret = "WORKER_TOKEN_MUST_NOT_REACH_CLIENT_OR_LOGS"  # pragma: allowlist secret -- synthetic test credential or sentinel only
        run_id = str(uuid.uuid4())
        with (
            patch.object(
                api_gateway,
                "_enqueue_with_backoff",
                return_value=(run_id, ["mock-job"]),
            ),
            patch.object(api_gateway, "WORKER_TRIGGER_MODE", "cloud_run_job"),
            patch.object(
                api_gateway,
                "_trigger_cloud_run_worker_once",
                side_effect=RuntimeError("upstream cloud diagnostics " + secret),
            ),
            self.assertLogs(api_gateway.LOGGER, level="INFO") as captured,
        ):
            response = self.client.post(
                "/v1/runs",
                headers=self.headers,
                json={"env": "tic_tac_toe", "model_names": ["MOCK:a", "MOCK:b"]},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["run_id"], run_id)
        self.assertIs(response.json()["worker_triggered"], False)
        self.assertTrue(response.json()["worker_trigger_error"])
        self.assertNotIn(secret, response.text)
        self.assertNotIn("upstream cloud diagnostics", response.text)
        logs = json.dumps([record.__dict__ for record in captured.records], default=str)
        self.assertNotIn(secret, logs)
        self.assertNotIn("upstream cloud diagnostics", logs)


class JevTransportSecurityTests(unittest.TestCase):
    def test_redirects_never_create_a_credential_forwarding_request(self):
        request = Request(jev.ENDPOINT, headers={"Authorization": "Bearer SENTINEL"})
        for status_code in (301, 302, 303, 307, 308):
            with self.subTest(status_code=status_code):
                self.assertIsNone(
                    jev._NoRedirect().redirect_request(
                        request,
                        io.BytesIO(b""),
                        status_code,
                        "Moved",
                        HTTPMessage(),
                        "https://attacker.invalid/collect",
                    )
                )

    def test_provider_response_size_is_bounded_before_json_parsing(self):
        response = MagicMock()
        response.read.return_value = b"x" * (jev.MAX_RESPONSE_BYTES + 1)
        response.__enter__.return_value = response
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(jev, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "size limit"):
                jev._post(Request(jev.ENDPOINT), 1)
        response.read.assert_called_once_with(jev.MAX_RESPONSE_BYTES + 1)

    def test_non_object_provider_json_is_rejected(self):
        response = MagicMock()
        response.read.return_value = json.dumps(["untrusted", "array"]).encode()
        response.__enter__.return_value = response
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(jev, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "must be an object"):
                jev._post(Request(jev.ENDPOINT), 1)

    def test_header_injection_is_rejected_before_transport(self):
        for credential in ("prefix\rInjected: secret", "prefix\nInjected: secret"):
            with self.subTest(credential=credential), self.assertRaises(ValueError):
                jev.JevModel(api_key=credential, mock_mode=False)


if __name__ == "__main__":
    unittest.main()
