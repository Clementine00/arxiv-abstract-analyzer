from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_ABSTRACT = (
    "We propose a novel deep learning architecture for image recognition "
    "that combines convolutional neural networks with attention mechanisms. "
    "Our model achieves state-of-the-art results on ImageNet and CIFAR-100 "
    "benchmarks, outperforming existing methods by a significant margin."
)

# --- Health check ---
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

# --- /classify tests ---
def test_classify_returns_200():
    response = client.post("/classify", json={"abstract": VALID_ABSTRACT})
    assert response.status_code == 200

def test_classify_has_expected_fields():
    response = client.post("/classify", json={"abstract": VALID_ABSTRACT})
    data = response.json()
    assert "predicted_category" in data
    assert "confidence" in data
    assert "all_scores" in data

def test_classify_confidence_between_0_and_1():
    response = client.post("/classify", json={"abstract": VALID_ABSTRACT})
    data = response.json()
    assert 0 <= data["confidence"] <= 1

def test_classify_all_categories_present():
    response = client.post("/classify", json={"abstract": VALID_ABSTRACT})
    data = response.json()
    assert len(data["all_scores"]) == 11

# --- /summarize tests ---
def test_summarize_returns_200():
    response = client.post("/summarize", json={"abstract": VALID_ABSTRACT})
    assert response.status_code == 200

def test_summarize_shorter_than_original():
    response = client.post("/summarize", json={"abstract": VALID_ABSTRACT})
    data = response.json()
    assert data["summary_length"] <= data["original_length"]

# --- Validation tests ---
def test_short_abstract_rejected():
    response = client.post("/classify", json={"abstract": "Too short."})
    assert response.status_code == 422

def test_empty_abstract_rejected():
    response = client.post("/classify", json={"abstract": ""})
    assert response.status_code == 422

def test_missing_abstract_rejected():
    response = client.post("/classify", json={})
    assert response.status_code == 422