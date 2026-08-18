# TryTrek — AI Real-Time Virtual Fitting Room
---

### ⏱️ Overview:
1. **Live Camera Feed**: Standard webcam input processed at 30+ FPS via persistent video stream manager.
2. **Pose Landmark Alignment**: MediaPipe tracks 33 3D body points to compute shoulder width and torso height dynamically.
3. **Instant Try-On**: Select any shirt from the UI grid. A lightweight REST call (`/select_clothing`) swaps garments in RAM without stopping or restarting the stream.
4. **Alpha Matrix Compositing**: Custom OpenCV + NumPy linear interpolation overlays semi-transparent PNG garments smoothly onto torso coordinates.

---

## 📐 2. Architecture Diagram

```
                ┌─────────────┐
                │   Webcam    │
                └──────┬──────┘
                       ↓
                ┌─────────────┐
                │   OpenCV    │
                └──────┬──────┘
                       ↓
              ┌─────────────────┐
              │ MediaPipe Pose  │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Clothing Fitter │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Alpha Blending  │
              └────────┬────────┘
                       ↓
                 FastAPI/MJPEG
```

---

## 🧪 3. Comprehensive Test Suite

TryTrek includes a production-grade automated test suite built with `pytest` and `fastapi.testclient.TestClient`.

### Covered Scenarios:
- ✅ **Clothing Selection**: Valid item selection via `POST` & `GET` `/select_clothing`.
- ❌ **Invalid Clothing Handling**: Returns `400 Bad Request` with descriptive error payload for non-existent items.
- 👤 **Pose Detection Fallback**: Robust frame compositing fallback when pose landmarks are out of bounds or missing.
- 📡 **API Endpoints**: Full verification of `/`, `/video_feed` (multipart MJPEG stream), and `/toggle_skeleton`.
- 📷 **Camera Initialization**: Singleton `CameraStreamManager` thread-safe initialization test.
- ⚡ **Concurrent Requests**: High-concurrency stress testing (30+ parallel threads) for state lock safety under load.

### Run Tests:
```bash
pytest -v tests/
```

---

## 🔄 4. Continuous Integration (CI/CD Pipeline)

Automated GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every `push` and `pull_request`:

```
push / PR
   ↓
[ Step 1: Linting Check ] ------- (Flake8 syntax & style enforcement)
   ↓
[ Step 2: Unit & Integration ] - (Pytest test suite)
   ↓
[ Step 3: Build Verification ] -- (Python bytecode compilation)
   ↓
[ Step 4: Security Scan ] ------- (Bandit vulnerability analysis)
```

---

## ⚡ 5. Engineering Performance Benchmark

By replacing repeated OpenCV DirectShow camera driver re-initialization with a persistent **Singleton Camera Manager**, cold-start startup delay was eliminated:

| Metric | Before Optimization | After Optimization | Engineering Impact |
|---|---|---|---|
| **Camera Initialization** | `5.02 s` (5020 ms) | `42 ms` | **~99.2% Faster** 🚀 |
| **Garment Switching Latency** | Direct stream restart (~3-5s) | `0.4 ms` (REST RAM swap) | **Instant / Zero Drop** |
| **Alpha Compositing Speed** | ~12.5 ms / frame | `1.2 ms` / frame | **800+ FPS Capacity** |

### Run Benchmark Suite:
```bash
python benchmark.py
```

---

## 🚀 Features

- **33 3D Landmark Tracking**: MediaPipe Pose extracts accurate shoulder & hip coordinates.
- **Zero-Stream-Interruption Swaps**: REST API state update protected by thread-safe `threading.Lock()`.
- **Pre-loaded Asset Cache**: All clothing RGBA PNGs pre-cached into RAM at server boot up.
- **Pose Guide Overlay**: Interactive alignment skeleton guide toggle (`/toggle_skeleton`).
- **Clean Responsive UI**: Modern CSS grid dashboard for real-time virtual fitting.

---

## 🛠️ Tech Stack & Dependencies

| Package | Version | Purpose in Codebase |
|---|---|---|
| **`fastapi`** | `^0.110.0` | Async web framework & REST API routes |
| **`uvicorn`** | `^0.28.0` | High-performance ASGI web server |
| **`mediapipe`** | `^0.10.14` | ML framework detecting 33 body pose landmarks |
| **`opencv-python`** | `^4.9.0` | Camera frame capture & matrix rendering |
| **`numpy`** | `^1.26.0` | Fast matrix alpha-channel blending |
| **`pytest`** | `^8.0.0` | Automated test suite execution |
| **`bandit`** | `^1.7.8` | AST-based security vulnerability auditing |

---

## ⚙️ Quick Start

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/gautambandil08/Virtual-Fitting-Room.git
cd Virtual-Fitting-Room
pip install -r requirements.txt
```

### 2. Run Main Application
```bash
python main.py
# Access dashboard at: http://127.0.0.1:8000
```

### 3. Run Verification & Benchmarks
```bash
# Run test suite
pytest -v tests/

# Run benchmark script
python benchmark.py
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main interactive virtual fitting mirror studio UI |
| `GET` | `/video_feed` | Live MJPEG camera & clothing overlay stream |
| `POST` | `/select_clothing?clothing=Blue-t-shirt` | Switch garment live without stream drop |
| `POST` | `/toggle_skeleton` | Toggle MediaPipe pose skeleton guide overlay |

---

## 👨‍💻 Author

**Gautam Bandil**  
- GitHub: [@gautambandil08](https://github.com/gautambandil08)  
- Project Repository: [Virtual-Fitting-Room](https://github.com/gautambandil08/Virtual-Fitting-Room)
