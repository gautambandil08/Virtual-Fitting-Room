# TryTrek — AI Real-Time Virtual Fitting Room

> **Real-time AI-powered virtual clothing try-on** using **MediaPipe Pose**, **OpenCV**, and **FastAPI** with a clean humanized web interface.

---

## ✨ Demo

> 📹 *[Add your YouTube demo video link here]*
> 
> 🔗 *[Add your live public tunnel link here (localtunnel / ngrok)]*

---

## 🚀 Features

- **Real-time body pose detection** using MediaPipe's 33 3D skeletal landmarks
- **Zero-latency clothing overlay switching** via custom REST API `/select_clothing` — no camera restart, no stream interruption
- **Alpha-channel matrix blending** (NumPy) for realistic transparent clothing compositing
- **Persistent camera + ML model manager** — eliminates Windows DirectShow driver re-initialization lag (~5s → <50ms)
- **MJPEG multi-part streaming** via FastAPI `StreamingResponse`
- **Pose skeleton alignment guide toggle** (`/toggle_skeleton` endpoint)

---

## 🛠️ Tech Stack & Dependencies

Strictly 5 core production dependencies:

| Package | Purpose in Codebase |
|---|---|
| **`fastapi`** | Async web server framework & REST API endpoints (`/select_clothing`, `/toggle_skeleton`) |
| **`uvicorn`** | ASGI server implementation running `main.py` |
| **`mediapipe`** | Google ML framework detecting 33 body pose landmarks in real time |
| **`opencv-python`** | Frame capture (`cv2.VideoCapture`), image manipulation, JPEG encoding |
| **`numpy`** | Array operations & alpha-channel linear interpolation for clothing overlay |

---

## 📁 Project Structure

```
trytrek/
├── static/                     # Image assets directory
│   ├── boys_11.png            # Classic White Crewneck (RGBA)
│   ├── boys_22.png            # Royal Blue Athletic Tee (RGBA)
│   ├── boys_33.png            # Stealth Black Tee (RGBA)
│   ├── boys_44.png            # Crimson Red Graphic Tee (RGBA)
│   ├── boys_55.png            # Emerald Green Casual Tee (RGBA)
│   └── boys_66.png            # Neon Yellow Vibe Tee (RGBA)
├── main.py                     # FastAPI backend — streaming, REST API, pose detection
├── virtual_try_on.py           # Standalone OpenCV module (no server)
├── requirements.txt            # Minimal 5-package dependency manifest
├── README.md                   # Recruiter-facing documentation
└── .gitignore                  # Git exclusion rules
```

---

## ⚙️ How It Works

```
Webcam → OpenCV Frame → MediaPipe Pose → 33 Landmark Detection
    → Shoulder/Hip width & height calculation
    → NumPy affine resize & position of clothing PNG
    → Alpha-channel blending onto frame
    → MJPEG JPEG encode → FastAPI StreamingResponse → Browser
```

When you click a different outfit:
```
Frontend "Try On" → POST /select_clothing → global state update (thread-safe lock)
    → Next frame in stream uses new clothing instantly (no stream reset)
```

---

## 🔧 Setup & Run

### Prerequisites
- Python 3.10+
- Webcam connected

### Install dependencies (5 packages)
```bash
pip install -r requirements.txt
```

### Run FastAPI application
```bash
python main.py
# Open: http://127.0.0.1:8000
```

### Get a public shareable link (localtunnel)
```bash
# Install localtunnel (once)
npm install -g localtunnel

# While main.py is running:
lt --port 8000
# Copy the https://xxxx.loca.lt link
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main web interface |
| `GET` | `/video_feed` | Live MJPEG camera stream |
| `POST` | `/select_clothing?clothing=Blue-t-shirt` | Switch outfit instantly |
| `POST` | `/toggle_skeleton` | Toggle pose landmark debug overlay |

---

## 🧠 Key Engineering Decisions

1. **Singleton Camera Manager** — One `cv2.VideoCapture` instance shared across stream requests prevents multi-second DirectShow re-initialization delays on Windows.
2. **Pre-loaded RGBA Assets** — All 6 shirt PNGs in `static/` loaded into RAM at startup for zero disk I/O during streaming frames.
3. **Thread-safe Global State** — `threading.Lock()` protects `current_selected_clothing` from race conditions between HTTP requests.
4. **REST over WebSocket** — `/select_clothing` uses a lightweight POST pattern instead of WebSockets, keeping infrastructure simple and light.

---

## 🙋 Author

**[Your Name]**  
📧 [your.email@gmail.com]  
🔗 [linkedin.com/in/yourprofile]  
💻 [github.com/yourusername]
