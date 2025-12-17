#!/usr/bin/env python3
"""
Test script for VLA API endpoints
"""
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
import base64
import io
from PIL import Image

client = TestClient(app)

def create_test_image():
    """Create a test image and convert to base64"""
    # Create a simple test image
    image_array = (np.random.random((224, 224, 3)) * 255).astype(np.uint8)
    image = Image.fromarray(image_array)

    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def test_process_vla():
    """Test VLA processing endpoint"""
    print("Testing VLA Processing Endpoint...")

    image_data = create_test_image()
    request_data = {
        "image": image_data,
        "instruction": "pick up the red object",
        "fusion_method": "cross_attention"
    }

    response = client.post("/api/vla/process", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Action type: {response_data.get('action_type')}")
        print(f"Confidence: {response_data.get('confidence')}")
        assert response_data["success"] == True
        assert "action_type" in response_data
        print("PASS: VLA Processing test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: VLA Processing test failed with status {response.status_code}\n")


def test_analyze_vision():
    """Test vision analysis endpoint"""
    print("Testing Vision Analysis Endpoint...")

    image_data = create_test_image()
    request_data = {
        "image": image_data,
        "query": "red object"
    }

    response = client.post("/api/vla/analyze-vision", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        assert response_data["success"] == True
        assert "object_detections" in response_data
        print(f"Objects detected: {len(response_data['object_detections'])}")
        print("PASS: Vision Analysis test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Vision Analysis test failed with status {response.status_code}\n")


def test_analyze_language():
    """Test language analysis endpoint"""
    print("Testing Language Analysis Endpoint...")

    request_data = {
        "text": "Grasp the blue cup and place it on the table"
    }

    response = client.post("/api/vla/analyze-language", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        assert response_data["success"] == True
        assert "intent" in response_data
        print(f"Detected intent: {response_data['intent']}")
        print(f"Entities found: {len(response_data['entities'])}")
        print("PASS: Language Analysis test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Language Analysis test failed with status {response.status_code}\n")


def test_fuse_multimodal():
    """Test multimodal fusion endpoint"""
    print("Testing Multimodal Fusion Endpoint...")

    request_data = {
        "vision_features": [float(x) for x in np.random.random(256)],
        "language_features": [float(x) for x in np.random.random(256)],
        "fusion_method": "cross_attention"
    }

    response = client.post("/api/vla/fuse-multimodal", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        assert response_data["success"] == True
        assert "fused_features" in response_data
        print(f"Fused features length: {len(response_data['fused_features'])}")
        print("PASS: Multimodal Fusion test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Multimodal Fusion test failed with status {response.status_code}\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting VLA API Tests\n")

    test_process_vla()
    test_analyze_vision()
    test_analyze_language()
    test_fuse_multimodal()

    print("All VLA API tests completed!")


if __name__ == "__main__":
    run_api_tests()