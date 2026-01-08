import httpx

def test_health():
    r = httpx.get("http://localhost:8000/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
