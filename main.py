from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
from typing import Optional

app = FastAPI(title="TryTrek | Virtual Fitting Mirror")

# Mount static directory to serve images
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Virtual clothing items mapping
clothing_items = {
    'White-t-shirt': 'boys_11.png',
    'Blue-t-shirt': 'boys_22.png',
    'Black-t-shirt': 'boys_33.png',
    'Red-t-shirt': 'boys_44.png',
    'Green-t-shirt': 'boys_55.png',
    'Yellow-t-shirt': 'boys_66.png'
}

clothing_labels = {
    'White-t-shirt': 'Classic White Crewneck',
    'Blue-t-shirt': 'Royal Blue Athletic Tee',
    'Black-t-shirt': 'Stealth Black Tee',
    'Red-t-shirt': 'Crimson Red Graphic Tee',
    'Green-t-shirt': 'Emerald Green Casual Tee',
    'Yellow-t-shirt': 'Neon Yellow Vibe Tee'
}

# Pre-load clothing images into memory for instant rendering
preloaded_clothing = {}
for key, filename in clothing_items.items():
    filepath = os.path.join(static_dir, filename)
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is not None:
        preloaded_clothing[key] = img
        print(f"Pre-loaded {key} ({filename}) shape={img.shape}")
    else:
        print(f"Warning: Could not load {filepath}")

# Global state for current selected clothing, skeleton mode, and thread locks
current_selected_clothing = "White-t-shirt"
show_skeleton = False
state_lock = threading.Lock()

class CameraStreamManager:
    def __init__(self):
        self.cap = None
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.lock = threading.Lock()

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)

    def get_frame(self, clothing_key, draw_skeleton=False):
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                self.open_camera()
            
            success, image = self.cap.read()
            if not success:
                self.open_camera()
                success, image = self.cap.read()
                if not success:
                    return None

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(image_rgb)

            clothing = preloaded_clothing.get(clothing_key)
            if pose_results.pose_landmarks:
                if clothing is not None:
                    try:
                        resized_clothing = resize_clothing(image, clothing, pose_results.pose_landmarks.landmark)
                        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        shoulder_x = int((left_shoulder.x + right_shoulder.x) / 2 * image.shape[1])
                        shoulder_y = int((left_shoulder.y + right_shoulder.y) / 2 * image.shape[0])
                        offset_y = 200
                        clothing_position = (shoulder_x - resized_clothing.shape[1] // 2,
                                             shoulder_y - resized_clothing.shape[0] // 2 + offset_y)
                        image = overlay_clothing(image, resized_clothing, clothing_position)
                    except Exception as e:
                        print("Error overlaying clothing:", e)
                
                if draw_skeleton:
                    mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            _, buffer = cv2.imencode('.jpg', image)
            return buffer.tobytes()

camera_manager = CameraStreamManager()

def resize_clothing(image, clothing, landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    shoulder_width = np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - 
                                    np.array([right_shoulder.x, right_shoulder.y])) * image.shape[1]
    torso_height = np.linalg.norm(np.array([(left_shoulder.x + right_shoulder.x)/2, (left_shoulder.y + right_shoulder.y)/2]) - 
                                  np.array([(left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2])) * image.shape[0]
    width_scale = shoulder_width / clothing.shape[1] * 2.2
    height_scale = torso_height / clothing.shape[0] * 1.6
    new_size = (int(clothing.shape[1] * width_scale), int(clothing.shape[0] * height_scale))
    return cv2.resize(clothing, new_size)

def overlay_clothing(image, clothing, position):
    h, w = clothing.shape[:2]
    x, y = position
    if x >= image.shape[1] or y >= image.shape[0] or x + w <= 0 or y + h <= 0:
        return image
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(image.shape[1], x + w)
    y_end = min(image.shape[0], y + h)
    clothing_x_start = x_start - x
    clothing_y_start = y_start - y
    clothing_x_end = clothing_x_start + (x_end - x_start)
    clothing_y_end = clothing_y_start + (y_end - y_start)
    
    if clothing.shape[2] == 4:
        alpha_s = clothing[clothing_y_start:clothing_y_end, clothing_x_start:clothing_x_end, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            image[y_start:y_end, x_start:x_end, c] = (alpha_s * clothing[clothing_y_start:clothing_y_end, clothing_x_start:clothing_x_end, c] +
                                                      alpha_l * image[y_start:y_end, x_start:x_end, c])
    else:
        image[y_start:y_end, x_start:x_end] = clothing[clothing_y_start:clothing_y_end, clothing_x_start:clothing_x_end]
    
    return image

def generate_frames():
    while True:
        with state_lock:
            active_clothing = current_selected_clothing
            active_skeleton = show_skeleton
        
        frame_bytes = camera_manager.get_frame(active_clothing, draw_skeleton=active_skeleton)
        if frame_bytes is None:
            time.sleep(0.03)
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

@app.post("/select_clothing")
@app.get("/select_clothing")
async def select_clothing(clothing: str = Query(...)):
    global current_selected_clothing
    if clothing in clothing_items or clothing in preloaded_clothing:
        with state_lock:
            current_selected_clothing = clothing
        return {"status": "ok", "selected_clothing": clothing}
    return JSONResponse(status_code=400, content={"error": f"Invalid clothing item '{clothing}'"})

@app.post("/toggle_skeleton")
@app.get("/toggle_skeleton")
async def toggle_skeleton():
    global show_skeleton
    with state_lock:
        show_skeleton = not show_skeleton
        status = show_skeleton
    return {"status": "ok", "show_skeleton": status}

@app.get("/video_feed")
async def video_feed(selected_clothing: Optional[str] = None):
    global current_selected_clothing
    if selected_clothing and (selected_clothing in clothing_items or selected_clothing in preloaded_clothing):
        with state_lock:
            current_selected_clothing = selected_clothing
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TryTrek | Virtual Fitting Mirror</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --border-color: #e2e8f0;
            --border-active: #4f46e5;
            --accent-primary: #4f46e5;
            --accent-primary-hover: #4338ca;
            --accent-light: #eef2ff;
            --accent-success: #059669;
            --text-title: #0f172a;
            --text-body: #475569;
            --text-muted: #64748b;
            --radius-lg: 20px;
            --radius-md: 14px;
            --radius-sm: 10px;
            --shadow-soft: 0 10px 30px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.02);
            --shadow-active: 0 12px 35px rgba(79, 70, 229, 0.15);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-page);
            color: var(--text-title);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Top Clean Header */
        header {
            background: #ffffff;
            border-bottom: 1px solid var(--border-color);
            padding: 18px 48px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            background: var(--accent-light);
            border: 1px solid rgba(79, 70, 229, 0.2);
            color: var(--accent-primary);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: 700;
        }

        .brand-title {
            font-size: 20px;
            font-weight: 800;
            color: var(--text-title);
            letter-spacing: -0.4px;
        }

        .brand-subtitle {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: var(--accent-success);
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 13px;
            font-weight: 600;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-success);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px rgba(5, 150, 105, 0.5);
        }

        /* Main Workspace Container */
        .workspace {
            max-width: 1380px;
            margin: 0 auto;
            width: 100%;
            padding: 36px 48px;
            display: grid;
            grid-template-columns: 1.35fr 1fr;
            gap: 36px;
            flex: 1;
        }

        /* Mirror Studio Container (Left) */
        .mirror-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 28px;
            box-shadow: var(--shadow-soft);
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-title);
        }

        .current-selection-pill {
            background: var(--accent-light);
            color: var(--accent-primary);
            font-size: 12px;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid rgba(79, 70, 229, 0.15);
        }

        .viewport-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            background: #0f172a;
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid var(--border-color);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }

        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .live-tag {
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(8px);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            color: var(--text-title);
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .controls-row {
            display: flex;
            gap: 12px;
        }

        .btn-toggle-guide {
            flex: 1;
            padding: 14px 20px;
            background: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            color: var(--text-title);
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s;
        }

        .btn-toggle-guide:hover {
            background: var(--accent-light);
            border-color: rgba(79, 70, 229, 0.3);
            color: var(--accent-primary);
        }

        .btn-toggle-guide.active {
            background: var(--accent-primary);
            color: #ffffff;
            border-color: var(--accent-primary);
        }

        /* Wardrobe Collection Side (Right) */
        .wardrobe-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 28px;
            box-shadow: var(--shadow-soft);
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .wardrobe-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            max-height: 520px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .wardrobe-grid::-webkit-scrollbar {
            width: 6px;
        }

        .wardrobe-grid::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 10px;
        }

        .outfit-item {
            background: #ffffff;
            border: 1.5px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }

        .outfit-item:hover {
            border-color: var(--accent-primary);
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.08);
        }

        .outfit-item.selected {
            border-color: var(--accent-primary);
            background: var(--accent-light);
            box-shadow: var(--shadow-active);
        }

        .img-container {
            width: 100%;
            height: 110px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 12px;
        }

        .outfit-thumb {
            max-height: 95px;
            max-width: 100%;
            object-fit: contain;
            transition: transform 0.25s;
        }

        .outfit-item:hover .outfit-thumb {
            transform: scale(1.06);
        }

        .outfit-name {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-title);
            margin-bottom: 12px;
            line-height: 1.3;
        }

        .btn-select {
            width: 100%;
            padding: 9px 14px;
            background: #f1f5f9;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            color: var(--text-title);
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .outfit-item.selected .btn-select {
            background: var(--accent-primary);
            color: #ffffff;
            border-color: var(--accent-primary);
        }

        .btn-select:hover {
            background: var(--accent-primary);
            color: #ffffff;
            border-color: var(--accent-primary);
        }

        @media (max-width: 1024px) {
            .workspace {
                grid-template-columns: 1fr;
                padding: 24px;
            }

            header {
                padding: 16px 24px;
            }
        }
    </style>
</head>

<body>

    <!-- Header -->
    <header>
        <div class="brand-container">
            <div class="brand-icon">✨</div>
            <div>
                <div class="brand-title">TryTrek Studio</div>
                <div class="brand-subtitle">Virtual Fitting Mirror</div>
            </div>
        </div>

        <div class="status-badge">
            <span class="status-dot"></span>
            <span>Virtual Mirror Ready</span>
        </div>
    </header>

    <!-- Main Workspace -->
    <main class="workspace">

        <!-- Left Column: Virtual Mirror -->
        <section class="mirror-card">
            <div class="section-header">
                <h2 class="section-title">Live Mirror View</h2>
                <div class="current-selection-pill" id="selected-label">Wearing: Classic White Crewneck</div>
            </div>

            <div class="viewport-wrapper">
                <img id="video-feed" class="video-feed" src="/video_feed" alt="Live Virtual Mirror Stream" />
                <div class="live-tag">
                    <span style="color: #ef4444;">●</span>
                    <span>Live Stream</span>
                </div>
            </div>

            <div class="controls-row">
                <button class="btn-toggle-guide" id="btn-guide" onclick="toggleGuide()">
                    <span>✨ Toggle Alignment Guide</span>
                </button>
            </div>
        </section>

        <!-- Right Column: Wardrobe Selection -->
        <section class="wardrobe-card">
            <div class="section-header">
                <div>
                    <h2 class="section-title">Select an Outfit</h2>
                    <div style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">Click any shirt below to try it on live</div>
                </div>
            </div>

            <div class="wardrobe-grid" id="wardrobe-grid">
                <!-- Wardrobe Items generated by JS -->
            </div>
        </section>

    </main>

    <!-- Frontend Logic -->
    <script>
        const wardrobe = {
            'White-t-shirt': { name: 'Classic White Crewneck', img: '/static/boys_11.png' },
            'Blue-t-shirt': { name: 'Royal Blue Athletic Tee', img: '/static/boys_22.png' },
            'Black-t-shirt': { name: 'Stealth Black Tee', img: '/static/boys_33.png' },
            'Red-t-shirt': { name: 'Crimson Red Graphic Tee', img: '/static/boys_44.png' },
            'Green-t-shirt': { name: 'Emerald Green Casual Tee', img: '/static/boys_55.png' },
            'Yellow-t-shirt': { name: 'Neon Yellow Vibe Tee', img: '/static/boys_66.png' }
        };

        let currentSelection = 'White-t-shirt';
        let isGuideActive = false;

        function renderWardrobe() {
            const container = document.getElementById('wardrobe-grid');
            container.innerHTML = '';

            Object.keys(wardrobe).forEach(key => {
                const item = wardrobe[key];
                const isSelected = (key === currentSelection);

                const card = document.createElement('div');
                card.className = `outfit-item ${isSelected ? 'selected' : ''}`;
                card.onclick = () => chooseOutfit(key);

                card.innerHTML = `
                    <div class="img-container">
                        <img src="${item.img}" alt="${item.name}" class="outfit-thumb">
                    </div>
                    <div class="outfit-name">${item.name}</div>
                    <button class="btn-select">
                        ${isSelected ? '✓ Wearing Now' : 'Try On'}
                    </button>
                `;

                container.appendChild(card);
            });
        }

        async function chooseOutfit(key) {
            currentSelection = key;
            document.getElementById('selected-label').textContent = `Wearing: ${wardrobe[key].name}`;
            renderWardrobe();

            try {
                await fetch(`/select_clothing?clothing=${key}`, { method: 'POST' });
            } catch (err) {
                console.error("Failed to select outfit:", err);
            }
        }

        async function toggleGuide() {
            isGuideActive = !isGuideActive;
            const btn = document.getElementById('btn-guide');

            if (isGuideActive) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }

            try {
                await fetch('/toggle_skeleton', { method: 'POST' });
            } catch (err) {
                console.error("Failed to toggle guide:", err);
            }
        }

        // Initialize UI
        renderWardrobe();
    </script>
</body>

</html>
""", status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
