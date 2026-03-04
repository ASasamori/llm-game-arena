import os
import unittest

from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("ARENA_API_KEYS", "local-dev-key")

from orchestrator.api_gateway import (
    ClientStylizeRequest,
    _render_stylized_client_payload,
    app,
)


class ClientStylizeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer local-dev-key"}

    def test_stylize_endpoint_returns_styled_payload(self) -> None:
        payload = {
            "title": "Client Demo",
            "summary": "This is a simple API request response.",
            "highlights": ["DB target selected", "Workers are long-lived"],
            "tone": "technical",
        }
        response = self.client.post(
            "/v1/client/stylize", headers=self.headers, json=payload
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["title"], "Client Demo")
        self.assertIn("Structured Highlights", body["styled_markdown"])
        self.assertIn("Summary:", body["styled_text"])

    def test_stylize_endpoint_requires_auth(self) -> None:
        payload = {"summary": "x"}
        response = self.client.post("/v1/client/stylize", json=payload)
        self.assertEqual(response.status_code, 401)

    def test_compact_renderer(self) -> None:
        markdown, text = _render_stylized_client_payload(
            ClientStylizeRequest(
                title="Compact",
                summary="One-liner summary",
                highlights=["A", "B"],
                tone="compact",
            )
        )
        self.assertIn("**Compact**", markdown)
        self.assertIn("Highlights:", text)


if __name__ == "__main__":
    unittest.main()
