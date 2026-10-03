from fastapi.testclient import TestClient


def make_client(app) -> TestClient:
    """A client that behaves like the UI page: local Host header and the CSRF header on every request."""
    return TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Amanuensis-UI": "1"})
