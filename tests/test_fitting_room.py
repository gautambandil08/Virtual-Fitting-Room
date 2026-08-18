"""
TryTrek Virtual Fitting Room - Automated Test Suite
===================================================
Tests clothing selection, invalid inputs, pose detection fallback,
API endpoints, camera manager initialization, and thread safety.
"""

import pytest
import numpy as np
import cv2
import threading
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient

from main import (
    app, 
    clothing_items, 
    preloaded_clothing, 
    overlay_clothing, 
    resize_clothing,
    CameraStreamManager,
    current_selected_clothing,
    show_skeleton,
    state_lock,
    camera_manager
)

client = TestClient(app)

# Dummy encoded JPEG frame for testing video stream without hardware webcam
_, DUMMY_JPEG = cv2.imencode('.jpg', np.zeros((100, 100, 3), dtype=np.uint8))
DUMMY_FRAME_BYTES = DUMMY_JPEG.tobytes()

@pytest.fixture(autouse=True)
def mock_camera_get_frame():
    """Mock camera_manager.get_frame to avoid hardware camera block during tests."""
    with patch.object(camera_manager, 'get_frame', return_value=DUMMY_FRAME_BYTES):
        yield

# ---------------------------------------------------------------------------
# 1. Clothing Selection Tests
# ---------------------------------------------------------------------------
def test_valid_clothing_selection_post():
    """Verify selecting a valid clothing item via POST updates the selection."""
    response = client.post("/select_clothing?clothing=Blue-t-shirt")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["selected_clothing"] == "Blue-t-shirt"

def test_valid_clothing_selection_get():
    """Verify selecting a valid clothing item via GET updates the selection."""
    response = client.get("/select_clothing?clothing=Black-t-shirt")
    assert response.status_code == 200
    data = response.json()
    assert data["selected_clothing"] == "Black-t-shirt"

# ---------------------------------------------------------------------------
# 2. Invalid Clothing Handling Tests
# ---------------------------------------------------------------------------
def test_invalid_clothing_selection():
    """Verify selecting an invalid clothing item returns 400 Bad Request."""
    response = client.post("/select_clothing?clothing=NonExistentShirt")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Invalid clothing item" in data["error"]

def test_empty_clothing_selection():
    """Verify empty clothing parameter handling."""
    response = client.post("/select_clothing?clothing=")
    assert response.status_code == 400

# ---------------------------------------------------------------------------
# 3. Pose Detection Failure / Overlay Edge Cases
# ---------------------------------------------------------------------------
def test_overlay_clothing_out_of_bounds():
    """Verify overlaying clothing completely outside frame bounds returns original frame."""
    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clothing = np.zeros((100, 100, 4), dtype=np.uint8)
    
    # Position far out of bounds (top-left outside)
    result = overlay_clothing(base_frame.copy(), clothing, (-500, -500))
    np.testing.assert_array_equal(result, base_frame)
    
    # Position far out of bounds (bottom-right outside)
    result_far = overlay_clothing(base_frame.copy(), clothing, (1000, 1000))
    np.testing.assert_array_equal(result_far, base_frame)

def test_overlay_clothing_rgb_and_rgba():
    """Verify overlay handles both 3-channel RGB and 4-channel RGBA clothing."""
    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 4-channel RGBA
    clothing_rgba = np.ones((50, 50, 4), dtype=np.uint8) * 255
    res_rgba = overlay_clothing(base_frame.copy(), clothing_rgba, (10, 10))
    assert res_rgba.shape == (480, 640, 3)
    
    # 3-channel RGB
    clothing_rgb = np.ones((50, 50, 3), dtype=np.uint8) * 200
    res_rgb = overlay_clothing(base_frame.copy(), clothing_rgb, (10, 10))
    assert res_rgb.shape == (480, 640, 3)

# ---------------------------------------------------------------------------
# 4. API Endpoints Integration Tests
# ---------------------------------------------------------------------------
def test_index_endpoint():
    """Verify main index HTML page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "TryTrek" in response.text
    assert "Virtual Fitting Mirror" in response.text

def test_toggle_skeleton_endpoint():
    """Verify skeleton guide toggle endpoint."""
    response1 = client.post("/toggle_skeleton")
    assert response1.status_code == 200
    state1 = response1.json()["show_skeleton"]

    response2 = client.post("/toggle_skeleton")
    assert response2.status_code == 200
    state2 = response2.json()["show_skeleton"]

    assert state1 != state2

def test_video_feed_headers():
    """Verify video feed streaming response content-type."""
    def mock_gen():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + DUMMY_FRAME_BYTES + b'\r\n')

    with patch("main.generate_frames", side_effect=mock_gen):
        with client.stream("GET", "/video_feed") as response:
            assert response.status_code == 200
            assert "multipart/x-mixed-replace" in response.headers.get("content-type", "")
            for chunk in response.iter_bytes():
                if chunk:
                    assert len(chunk) > 0
                    break

# ---------------------------------------------------------------------------
# 5. Camera Manager Initialization Tests
# ---------------------------------------------------------------------------
def test_camera_manager_initialization():
    """Verify CameraStreamManager instance creation and lock properties."""
    manager = CameraStreamManager()
    assert manager.cap is None
    assert manager.lock is not None
    assert manager.pose is not None

# ---------------------------------------------------------------------------
# 6. Concurrent Requests & Thread Safety Tests
# ---------------------------------------------------------------------------
def test_concurrent_clothing_selection_and_toggles():
    """Verify state consistency during high-concurrency requests."""
    clothing_keys = list(clothing_items.keys())

    def worker(i):
        cloth = clothing_keys[i % len(clothing_keys)]
        r1 = client.post(f"/select_clothing?clothing={cloth}")
        r2 = client.post("/toggle_skeleton")
        return r1.status_code == 200 and r2.status_code == 200

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(20)]
        results = [f.result() for f in futures]

    assert all(results)
