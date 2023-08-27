from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# Test for the home endpoint
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "health_check": "OK",
        "model_version": "0.1.0",
    }  # Replace "0.1.0" with the actual model_version if it's different


# Test for the get_answer endpoint
def test_get_answer():
    response = client.post(
        "/get_answer",
        json={
            "video_url": "https://www.youtube.com/watch?v=vRVVyl9uaZc",
            "question": "What did he say about data classes in Python?",
        },
    )
    assert response.status_code == 200
