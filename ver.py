import inspect
# --- Patch for DeepFace/keras inspect.ArgSpec bug on Python 3.11+ ---
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
if not hasattr(inspect, 'ArgSpec'):
    from collections import namedtuple
    ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
    def _getargspec(func):
        fspec = inspect.getfullargspec(func)
        return ArgSpec(fspec.args, fspec.varargs, fspec.varkw, fspec.defaults)
    inspect.ArgSpec = ArgSpec
    inspect.getargspec = _getargspec
# --- End patch ---

import cv2
import os
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from PIL import Image, ImageTk
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Tkinter imports
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import queue

# === GPU SETUP ===
def setup_gpu():
    """Thiết lập GPU cho TensorFlow"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Đã phát hiện {len(gpus)} GPU(s)")
                print(f"   GPU devices: {[gpu.name for gpu in gpus]}")
                return True
            except RuntimeError as e:
                print(f"❌ Lỗi khi thiết lập GPU: {e}")
                return False
        else:
            print("⚠️  Không tìm thấy GPU, sẽ sử dụng CPU")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra GPU: {e}")
        return False

# === CONFIG ===
CAMERA_INDEX = 0
FRAME_SAVE_ROOT = Path("D:/fpt/subject/CPV301m/cpv/camera_capture_frames")
ALIGN_SAVE_ROOT = Path("D:/fpt/subject/CPV301m/cpv/Align")
EMBEDDED_DIR = Path("D:/fpt/subject/CPV301m/cpv/Embedded")
ATTENDANCE_LOG_PATH = Path("D:/fpt/subject/CPV301m/cpv/logs/attendance.csv")

# Model configuration
MODEL_NAME = "Facenet512"  # Có thể đổi: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace, SFace
THRESHOLD = 0.25

# === KHỞI TẠO MODEL MỘT LẦN ===
print(f"🔧 Đang khởi tạo model {MODEL_NAME}...")
try:
    from deepface.commons import functions
    functions.loadModel()
    
    test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    cv2.imwrite("test_init.jpg", test_img)
    _ = DeepFace.represent(
        img_path="test_init.jpg",
        model_name=MODEL_NAME,
        enforce_detection=False,
        detector_backend="skip"
    )
    os.remove("test_init.jpg")
    print(f"✅ Model {MODEL_NAME} đã sẵn sàng!")
except Exception as e:
    print(f"⚠️ Cảnh báo khi khởi tạo model: {e}")
    print("   Hệ thống sẽ thử khởi tạo lại khi cần...")

# === CORE FUNCTIONS ===
def load_logged_names(path):
    logged = set()
    if path.exists():
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 1:
                    logged.add(row[0])
    return logged

def capture_frames(name, user_id, log_callback=None):
    if log_callback:
        log_callback(f"🎥 Bắt đầu chụp khung hình cho: {name} (ID: {user_id})", "info")
    
    user_tag = f"{name}_{user_id}".replace(" ", "_")
    user_frame_dir = FRAME_SAVE_ROOT / user_tag
    user_frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        if log_callback:
            log_callback("❌ Lỗi: Không thể mở camera", "error")
        return

    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(round(camera_fps / 5)) if camera_fps and camera_fps > 0 else 6

    frame_count = 0
    saved_count = 0

    if log_callback:
        log_callback("📸 Đang chụp khung hình... (Nhấn 'Q' để dừng)", "info")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            frame_filename = f"frame_{saved_count+1:04d}.jpg"
            frame_path = user_frame_dir / frame_filename
            success, encoded_img = cv2.imencode('.jpg', frame)
            if success:
                with open(frame_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                saved_count += 1
                
                if saved_count % 10 == 0 and log_callback:
                    log_callback(f"📊 Đã lưu {saved_count} khung hình...", "info")

        frame_count += 1
        cv2.imshow("Capturing... Press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if log_callback:
        log_callback(f"✅ Hoàn thành: Đã lưu {saved_count} khung hình vào: {user_frame_dir}", "success")

def align_faces(log_callback=None):
    if log_callback:
        log_callback("🔄 Bắt đầu căn chỉnh khuôn mặt...", "info")
        log_callback("📋 Khởi tạo MTCNN detector...", "info")
    
    detector = MTCNN(device='cuda:0' if tf.config.list_physical_devices('GPU') else 'cpu')
    
    template = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    ALIGN_SAVE_ROOT.mkdir(exist_ok=True)
    
    total_folders = sum(1 for p in FRAME_SAVE_ROOT.iterdir() if p.is_dir())
    processed_folders = 0
    total_processed = 0
    total_failed = 0

    if log_callback:
        log_callback(f"📁 Tìm thấy {total_folders} thư mục để xử lý", "info")

    for person_folder in FRAME_SAVE_ROOT.iterdir():
        if not person_folder.is_dir():
            continue

        processed_folders += 1
        if log_callback:
            log_callback(f"👤 Đang xử lý [{processed_folders}/{total_folders}]: {person_folder.name}", "info")

        aligned_person_folder = ALIGN_SAVE_ROOT / person_folder.name
        aligned_person_folder.mkdir(exist_ok=True)

        images = list(person_folder.glob("*.jpg"))
        processed_images = 0
        failed_images = 0

        for i, img_path in enumerate(images):
            try:
                img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(img)
                
                if not detections:
                    failed_images += 1
                    continue

                keypoints = detections[0]['keypoints']
                src = np.array([
                    keypoints['left_eye'],
                    keypoints['right_eye'],
                    keypoints['nose'],
                    keypoints['mouth_left'],
                    keypoints['mouth_right']
                ], dtype=np.float32)

                M = cv2.estimateAffinePartial2D(src, template, method=cv2.LMEDS)[0]
                if M is None:
                    failed_images += 1
                    continue
                    
                aligned_face = cv2.warpAffine(img, M, (112, 112))
                save_path = aligned_person_folder / img_path.name
                Image.fromarray(aligned_face).save(save_path)
                processed_images += 1
                
                if (i + 1) % 20 == 0 and log_callback:
                    log_callback(f"📊 Đã xử lý {i + 1}/{len(images)} ảnh...", "info")
                    
            except Exception as e:
                failed_images += 1
                continue

        total_processed += processed_images
        total_failed += failed_images
        
        if log_callback:
            log_callback(f"✅ Hoàn thành: {processed_images} ảnh thành công, {failed_images} ảnh thất bại", "success")

    if log_callback:
        log_callback(f"🎉 Căn chỉnh khuôn mặt hoàn tất!", "success")
        log_callback(f"📊 Tổng kết: {total_processed} ảnh thành công, {total_failed} ảnh thất bại", "info")

def embed_faces(log_callback=None):
    if log_callback:
        log_callback("🧠 Bắt đầu tạo embedding cho khuôn mặt...", "info")
        log_callback(f"🔧 Sử dụng DeepFace với {MODEL_NAME} model...", "info")
    
    try:
        from deepface.commons import functions
        if hasattr(functions, 'models'):
            functions.models = {}
    except:
        pass
    
    EMBEDDED_DIR.mkdir(exist_ok=True)
    
    total_folders = sum(1 for p in ALIGN_SAVE_ROOT.iterdir() if p.is_dir())
    processed_folders = 0
    total_embeddings = 0

    if log_callback:
        log_callback(f"📁 Tìm thấy {total_folders} thư mục để xử lý", "info")

    for person_folder in ALIGN_SAVE_ROOT.iterdir():
        if not person_folder.is_dir():
            continue

        processed_folders += 1
        if log_callback:
            log_callback(f"👤 Đang tạo embedding [{processed_folders}/{total_folders}]: {person_folder.name}", "info")

        face_db = {}
        images = [f for f in os.listdir(person_folder) if f.endswith((".jpg", ".png"))]
        processed_images = 0
        failed_images = 0

        batch_size = 5
        
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            for i, file in enumerate(batch_images):
                img_path = os.path.join(person_folder, file)
                try:
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            embedding_obj = DeepFace.represent(
                                img_path=img_path,
                                model_name=MODEL_NAME,
                                enforce_detection=False,
                                detector_backend="skip"
                            )
                            
                            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                                embedding = embedding_obj[0]["embedding"]
                                face_db[f"{person_folder.name}_{file}"] = embedding
                                processed_images += 1
                                break
                            else:
                                raise ValueError("Empty embedding result")
                                
                        except Exception as retry_error:
                            if retry < max_retries - 1:
                                if log_callback:
                                    log_callback(f"🔄 Retry {retry + 1}/{max_retries} cho {file}", "warning")
                                tf.keras.backend.clear_session()
                                continue
                            else:
                                raise retry_error
                    
                    current_progress = batch_start + i + 1
                    if current_progress % 10 == 0 and log_callback:
                        log_callback(f"📊 Đã xử lý {current_progress}/{len(images)} ảnh...", "info")
                        
                except Exception as e:
                    failed_images += 1
                    if log_callback:
                        log_callback(f"⚠️ Không thể tạo embedding cho {file}: {str(e)[:50]}...", "warning")
                    continue

        if face_db:
            normalized_db = {}
            for k, v in face_db.items():
                try:
                    normalized_db[k] = normalize([v])[0]
                except:
                    if log_callback:
                        log_callback(f"⚠️ Không thể normalize embedding cho {k}", "warning")
                    
            if normalized_db:
                save_path = EMBEDDED_DIR / f"face_database_{person_folder.name}.npz"
                np.savez(save_path, **normalized_db)
                total_embeddings += len(normalized_db)
                
                if log_callback:
                    log_callback(f"✅ Hoàn thành: {processed_images} embedding thành công, {failed_images} ảnh thất bại", "success")
            else:
                if log_callback:
                    log_callback(f"❌ Không có embedding hợp lệ nào cho {person_folder.name}", "error")
        else:
            if log_callback:
                log_callback(f"❌ Không tạo được embedding nào cho {person_folder.name}", "error")

    if log_callback:
        log_callback(f"🎉 Tạo embedding hoàn tất!", "success")
        log_callback(f"📊 Tổng cộng: {total_embeddings} embeddings từ {processed_folders} người", "info")

def run_real_time_recognition(log_callback=None, stop_event=None):
    if log_callback:
        log_callback("🔍 Bắt đầu nhận dạng khuôn mặt thời gian thực...", "info")
        log_callback("📋 Đang tải database embeddings...", "info")

    detector = MTCNN(device='cuda:0' if tf.config.list_physical_devices('GPU') else 'cpu')
    db = {}

    embedding_files = list(EMBEDDED_DIR.glob("*.npz"))
    if not embedding_files:
        if log_callback:
            log_callback("❌ Không tìm thấy file embedding nào!", "error")
            log_callback("💡 Hãy chạy function 3 (Embed Faces) trước", "warning")
        return

    for npz_path in embedding_files:
        db_name = npz_path.stem.replace("face_database_", "")
        data = np.load(npz_path)
        for name in data.files:
            person_name = name.split('_frame_')[0]
            if person_name not in db:
                db[person_name] = []
            db[person_name].append(normalize([data[name]])[0])

    for person in db:
        if len(db[person]) > 1:
            db[person] = np.mean(db[person], axis=0)
        else:
            db[person] = db[person][0]

    if log_callback:
        log_callback(f"✅ Đã tải {len(db)} người từ database", "success")

    TEMPLATE = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    logged_names = load_logged_names(ATTENDANCE_LOG_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        if log_callback:
            log_callback("❌ Không thể mở camera!", "error")
        return

    if log_callback:
        log_callback("🎥 Camera đã sẵn sàng - Nhấn 'Q' để thoát", "success")

    frame_count = 0
    face_tracks = {}
    next_track_id = 0
    detection_interval = 5
    max_distance_threshold = 100
    max_frames_missing = 10
    smoothing_factor = 0.3

    fps_start_time = cv2.getTickCount()
    fps_frame_count = 0
    display_fps = 0

    while True:
        if stop_event and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_frame_count += 1

        if fps_frame_count >= 10:
            fps_end_time = cv2.getTickCount()
            time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            display_fps = fps_frame_count / time_diff
            fps_start_time = fps_end_time
            fps_frame_count = 0

        if frame_count % detection_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                results = detector.detect_faces(rgb)
                # --- START MODIFICATION ---
                if results:  # Nếu có bất kỳ khuôn mặt nào được tìm thấy
                    # Tìm khuôn mặt có diện tích bounding box lớn nhất (w * h)
                    largest_face = max(results, key=lambda face: face['box'][2] * face['box'][3])
                    # Chỉ giữ lại khuôn mặt lớn nhất để xử lý tiếp
                    results = [largest_face]
                # --- END MODIFICATION ---
            except:
                results = []

            current_detections = []

            for face in results:
                x, y, w, h = face['box']
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                if w <= 0 or h <= 0:
                    continue

                cx = x + w // 2
                cy = y + h // 2

                min_distance = float('inf')
                matched_track_id = None

                for track_id, track in face_tracks.items():
                    if track['missing_frames'] > 0:
                        continue

                    track_cx = track['x'] + track['w'] // 2
                    track_cy = track['y'] + track['h'] // 2
                    distance = np.sqrt((cx - track_cx)**2 + (cy - track_cy)**2)

                    if distance < min_distance and distance < max_distance_threshold:
                        min_distance = distance
                        matched_track_id = track_id

                if matched_track_id is not None:
                    track = face_tracks[matched_track_id]
                    track['x'] = int(track['x'] * (1 - smoothing_factor) + x * smoothing_factor)
                    track['y'] = int(track['y'] * (1 - smoothing_factor) + y * smoothing_factor)
                    track['w'] = int(track['w'] * (1 - smoothing_factor) + w * smoothing_factor)
                    track['h'] = int(track['h'] * (1 - smoothing_factor) + h * smoothing_factor)
                    track['missing_frames'] = 0
                    track['keypoints'] = face['keypoints']
                    track['confidence'] = face.get('confidence', 1.0)
                else:
                    face_tracks[next_track_id] = {
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'missing_frames': 0,
                        'keypoints': face['keypoints'],
                        'confidence': face.get('confidence', 1.0),
                        'name': None,
                        'similarity': 0.0,
                        'recognition_count': 0
                    }
                    next_track_id += 1

                current_detections.append((x, y, w, h))

            for track_id, track in list(face_tracks.items()):
                found = False
                track_cx = track['x'] + track['w'] // 2
                track_cy = track['y'] + track['h'] // 2

                for x, y, w, h in current_detections:
                    cx = x + w // 2
                    cy = y + h // 2
                    distance = np.sqrt((cx - track_cx)**2 + (cy - track_cy)**2)
                    if distance < max_distance_threshold:
                        found = True
                        break

                if not found:
                    track['missing_frames'] += 1
                    if track['missing_frames'] > max_frames_missing:
                        del face_tracks[track_id]

        for track_id, track in face_tracks.items():
            if track['missing_frames'] > 0:
                continue

            x, y, w, h = track['x'], track['y'], track['w'], track['h']

            if (frame_count % 15 == 0 or track['name'] is None) and 'keypoints' in track:
                keypoints = track['keypoints']
                src = np.array([
                    keypoints['left_eye'], keypoints['right_eye'],
                    keypoints['nose'], keypoints['mouth_left'], keypoints['mouth_right']
                ], dtype=np.float32)

                M, _ = cv2.estimateAffinePartial2D(src, TEMPLATE, method=cv2.LMEDS)
                if M is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    aligned = cv2.warpAffine(rgb, M, (112, 112))
                    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
                    temp_path = f"temp_face_{track_id}_{frame_count}.jpg"
                    cv2.imwrite(temp_path, aligned_bgr)

                    try:
                        result = DeepFace.represent(
                            img_path=temp_path,
                            model_name=MODEL_NAME,
                            detector_backend="skip",
                            enforce_detection=False
                        )

                        if result and isinstance(result, list) and len(result) > 0:
                            emb = normalize([result[0]["embedding"]])[0]

                            best_match = None
                            best_dist = float("inf")
                            for person_id, stored_emb in db.items():
                                dist = cosine(emb, stored_emb)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_match = person_id

                            similarity = 1 - best_dist

                            if track['name'] is None or track['recognition_count'] < 5:
                                if best_dist < THRESHOLD:
                                    track['name'] = best_match
                                    track['similarity'] = similarity
                                    track['recognition_count'] += 1
                                else:
                                    track['name'] = "Unknown"
                                    track['similarity'] = similarity
                            else:
                                track['similarity'] = track['similarity'] * 0.7 + similarity * 0.3

                    except Exception as e:
                        if log_callback:
                            log_callback(f"⚠️ Lỗi recognition: {str(e)[:50]}...", "warning")

                    finally:
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass

            if track.get('name'):
                if track['name'] != "Unknown":
                    label = f"{track['name']}"
                    color = (0, 255, 0)

                    if track['name'] not in logged_names and track['recognition_count'] >= 3:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ATTENDANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                        with open(ATTENDANCE_LOG_PATH, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([track['name'], now, "Present"])
                        logged_names.add(track['name'])
                        if log_callback:
                            log_callback(f"✅ Điểm danh: {track['name']} - {now}", "success")
                else:
                    label = "Unknown"
                    color = (0, 165, 255)
            else:
                label = "Detecting..."
                color = (255, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        active_faces = sum(1 for t in face_tracks.values() if t['missing_frames'] == 0)
        cv2.putText(frame, f"Faces: {active_faces}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    for f in Path(".").glob("temp_face_*.jpg"):
        try:
            os.remove(f)
        except:
            pass

    if log_callback:
        log_callback("🛑 Đã dừng nhận dạng thời gian thực", "warning")

# === UI CLASS ===
class AutoAttendUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoAttend - Hệ thống Điểm danh Thông minh")
        self.root.geometry("1400x850")
        
        # Set theme colors
        self.bg_color = "#0f0f0f"
        self.panel_bg = "#1a1a1a"
        self.fg_color = "#ffffff"
        self.accent_color = "#00ff88"
        self.success_color = "#4ade80"
        self.error_color = "#f87171"
        self.warning_color = "#fbbf24"
        self.button_bg = "#2563eb"
        self.button_hover = "#1d4ed8"
        
        self.root.configure(bg=self.bg_color)
        
        # Variables
        self.camera_active = False
        self.recognition_active = False
        self.recognition_thread = None
        self.recognition_stop_event = threading.Event()
        self.capture_thread = None
        self.current_frame = None
        self.cap = None
        self.recognition_overlay = None  # Lưu kết quả nhận diện (boxes, labels)
        self.overlay_lock = threading.Lock()
        
        # Setup GPU
        self.gpu_available = setup_gpu()
        
        # Create UI
        self.create_widgets()
        
        # Start camera preview
        self.update_camera_preview()
        
    def create_widgets(self):
        """Create main UI layout"""
        self.create_header()
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.create_control_panel(main_frame)
        self.create_camera_panel(main_frame)
        self.create_info_panel(main_frame)
        self.create_status_bar()
        
    def create_header(self):
        """Create application header"""
        header_frame = tk.Frame(self.root, bg=self.panel_bg, height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_frame = tk.Frame(header_frame, bg=self.panel_bg)
        title_frame.pack(side=tk.LEFT, padx=30, pady=15)
        
        logo_label = tk.Label(title_frame, text="🎯", font=("Arial", 36), 
                             bg=self.panel_bg, fg=self.accent_color)
        logo_label.pack(side=tk.LEFT, padx=(0, 15))
        
        title_label = tk.Label(title_frame, text="AutoAttend System", 
                              font=("Arial", 28, "bold"), 
                              bg=self.panel_bg, fg=self.fg_color)
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(title_frame, text="Hệ thống Điểm danh Thông minh với AI", 
                                 font=("Arial", 12), 
                                 bg=self.panel_bg, fg="#888888")
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
        gpu_frame = tk.Frame(header_frame, bg=self.panel_bg)
        gpu_frame.pack(side=tk.RIGHT, padx=30, pady=25)
        
        gpu_status = "GPU ✅" if self.gpu_available else "CPU ⚠️"
        gpu_label = tk.Label(gpu_frame, text=f"Status: {gpu_status}", 
                            font=("Arial", 12, "bold"),
                            bg=self.panel_bg, 
                            fg=self.success_color if self.gpu_available else self.warning_color)
        gpu_label.pack()
        
    def create_control_panel(self, parent):
        """Create left control panel"""
        control_frame = tk.Frame(parent, bg=self.panel_bg, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        header = tk.Label(control_frame, text="Control Panel", 
                         font=("Arial", 18, "bold"),
                         bg=self.panel_bg, fg=self.accent_color)
        header.pack(pady=20)
        
        input_frame = tk.LabelFrame(control_frame, text="📝 Thông tin người dùng", 
                                   font=("Arial", 12, "bold"),
                                   bg=self.panel_bg, fg=self.fg_color,
                                   labelanchor="n", pady=10, padx=10)
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(input_frame, text="Họ và tên:", font=("Arial", 11),
                bg=self.panel_bg, fg=self.fg_color).grid(row=0, column=0, sticky=tk.W, pady=8)
        self.name_var = tk.StringVar()
        self.name_entry = tk.Entry(input_frame, textvariable=self.name_var,
                                  font=("Arial", 11), width=20,
                                  bg="#2a2a2a", fg=self.fg_color,
                                  insertbackground=self.fg_color)
        self.name_entry.grid(row=0, column=1, pady=8, padx=(10, 0))
        
        tk.Label(input_frame, text="ID/MSSV:", font=("Arial", 11),
                bg=self.panel_bg, fg=self.fg_color).grid(row=1, column=0, sticky=tk.W, pady=8)
        self.id_var = tk.StringVar()
        self.id_entry = tk.Entry(input_frame, textvariable=self.id_var,
                                font=("Arial", 11), width=20,
                                bg="#2a2a2a", fg=self.fg_color,
                                insertbackground=self.fg_color)
        self.id_entry.grid(row=1, column=1, pady=8, padx=(10, 0))
        
        buttons_frame = tk.LabelFrame(control_frame, text="🚀 Chức năng chính",
                                     font=("Arial", 12, "bold"),
                                     bg=self.panel_bg, fg=self.fg_color,
                                     labelanchor="n", pady=10)
        buttons_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        button_style = {
            "font": ("Arial", 11, "bold"),
            "width": 25,
            "height": 2,
            "cursor": "hand2",
            "relief": tk.FLAT,
            "bd": 0
        }
        
        self.capture_btn = tk.Button(buttons_frame, text="📸 Chụp ảnh đăng ký",
                                    command=self.start_capture,
                                    bg="#10b981", fg="white",
                                    activebackground="#059669",
                                    **button_style)
        self.capture_btn.pack(pady=8, padx=20)
        
        self.align_btn = tk.Button(buttons_frame, text="🔄 Căn chỉnh khuôn mặt",
                                  command=self.run_align,
                                  bg="#3b82f6", fg="white",
                                  activebackground="#2563eb",
                                  **button_style)
        self.align_btn.pack(pady=8, padx=20)
        
        self.embed_btn = tk.Button(buttons_frame, text="🧠 Tạo Embedding",
                                  command=self.run_embed,
                                  bg="#8b5cf6", fg="white",
                                  activebackground="#7c3aed",
                                  **button_style)
        self.embed_btn.pack(pady=8, padx=20)
        
        self.recognize_btn = tk.Button(buttons_frame, text="🔍 Bắt đầu nhận dạng",
                                      command=self.toggle_recognition,
                                      bg="#ef4444", fg="white",
                                      activebackground="#dc2626",
                                      **button_style)
        self.recognize_btn.pack(pady=8, padx=20)
        
        extra_frame = tk.LabelFrame(control_frame, text="🛠️ Chức năng khác",
                                   font=("Arial", 12, "bold"),
                                   bg=self.panel_bg, fg=self.fg_color,
                                   labelanchor="n", pady=10)
        extra_frame.pack(fill=tk.X, padx=20)
        
        tk.Button(extra_frame, text="📊 Xuất báo cáo",
                 command=self.export_report,
                 bg="#059669", fg="white",
                 font=("Arial", 10), width=20, height=1,
                 cursor="hand2", relief=tk.FLAT).pack(pady=5)
        
        tk.Button(extra_frame, text="🗑️ Xóa dữ liệu",
                 command=self.clear_data,
                 bg="#dc2626", fg="white",
                 font=("Arial", 10), width=20, height=1,
                 cursor="hand2", relief=tk.FLAT).pack(pady=5)
        
    def create_camera_panel(self, parent):
        """Create center camera view panel"""
        camera_frame = tk.Frame(parent, bg=self.panel_bg)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        cam_header = tk.Frame(camera_frame, bg=self.panel_bg)
        cam_header.pack(fill=tk.X, pady=(20, 10))
        
        tk.Label(cam_header, text="📷 Camera View",
                font=("Arial", 18, "bold"),
                bg=self.panel_bg, fg=self.accent_color).pack(side=tk.LEFT, padx=20)
        
        self.cam_status_label = tk.Label(cam_header, text="● Live",
                                        font=("Arial", 12),
                                        bg=self.panel_bg, fg=self.success_color)
        self.cam_status_label.pack(side=tk.LEFT)
        
        self.camera_label = tk.Label(camera_frame, bg="#000000",
                                    text="Camera đang khởi động...",
                                    font=("Arial", 16), fg="#666666")
        self.camera_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=(0, 20))
        
    def create_info_panel(self, parent):
        """Create right information panel"""
        info_frame = tk.Frame(parent, bg=self.panel_bg, width=400)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)
        
        stats_frame = tk.LabelFrame(info_frame, text="📊 Thống kê",
                                   font=("Arial", 14, "bold"),
                                   bg=self.panel_bg, fg=self.accent_color,
                                   labelanchor="n", pady=10)
        stats_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        self.stats_labels = {}
        stats = [
            ("Tổng số người", "total_users", "0"),
            ("Đã điểm danh hôm nay", "attended_today", "0"),
            ("Đang online", "online_now", "0"),
            ("Tỷ lệ điểm danh", "attendance_rate", "0%")
        ]
        
        for i, (label, key, default) in enumerate(stats):
            frame = tk.Frame(stats_frame, bg=self.panel_bg)
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(frame, text=f"{label}:",
                    font=("Arial", 11), bg=self.panel_bg,
                    fg=self.fg_color).pack(side=tk.LEFT)
            
            self.stats_labels[key] = tk.Label(frame, text=default,
                                             font=("Arial", 11, "bold"),
                                             bg=self.panel_bg, fg=self.warning_color)
            self.stats_labels[key].pack(side=tk.RIGHT)
        
        log_frame = tk.LabelFrame(info_frame, text="📜 Nhật ký hoạt động",
                                 font=("Arial", 14, "bold"),
                                 bg=self.panel_bg, fg=self.accent_color,
                                 labelanchor="n")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        log_container = tk.Frame(log_frame, bg=self.panel_bg)
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(log_container, bg=self.panel_bg)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_container, wrap=tk.WORD,
                               font=("Consolas", 10),
                               bg="#0a0a0a", fg=self.fg_color,
                               yscrollcommand=scrollbar.set,
                               height=15)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        self.log_text.tag_config("success", foreground=self.success_color)
        self.log_text.tag_config("error", foreground=self.error_color)
        self.log_text.tag_config("warning", foreground=self.warning_color)
        self.log_text.tag_config("info", foreground=self.accent_color)
        
        self.add_log("Hệ thống AutoAttend đã sẵn sàng", "success")
        
    def create_status_bar(self):
        """Create bottom status bar"""
        status_frame = tk.Frame(self.root, bg="#0a0a0a", height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Ready",
                                    font=("Arial", 10),
                                    bg="#0a0a0a", fg=self.fg_color)
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.time_label = tk.Label(status_frame, font=("Arial", 10),
                                  bg="#0a0a0a", fg=self.fg_color)
        self.time_label.pack(side=tk.RIGHT, padx=20)
        self.update_time()
        
    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
        
    def add_log(self, message, tag="info"):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_text.insert(tk.END, f"{timestamp} ", "info")
        self.log_text.insert(tk.END, f"{message}\n", tag)
        self.log_text.see(tk.END)
        
    def update_camera_preview(self):
        """Update camera preview"""
        # Only update preview if not recognizing
        if self.recognition_active:
            self.root.after(30, self.update_camera_preview)
            return
        if not self.camera_active:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.camera_active = True
                    self.add_log("Camera đã được kích hoạt", "success")
        if self.camera_active and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.config(image=imgtk, text="")
                self.camera_label.image = imgtk
        self.root.after(30, self.update_camera_preview)

    def start_capture(self):
        """Start face capture process"""
        name = self.name_var.get().strip()
        user_id = self.id_var.get().strip()
        if not name or not user_id:
            messagebox.showwarning("Thiếu thông tin", 
                                 "Vui lòng nhập đầy đủ họ tên và ID!")
            return
        self.add_log(f"Bắt đầu chụp ảnh cho {name} (ID: {user_id})", "info")
        self.status_label.config(text=f"Đang chụp ảnh cho {name}...")
        def capture_and_release():
            capture_frames(name, user_id, self.add_log)
            # Release camera and reset state after capture
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.camera_active = False
            self.root.after(100, self.update_camera_preview)
        thread = threading.Thread(target=capture_and_release)
        thread.daemon = True
        thread.start()

    def run_align(self):
        """Run face alignment process"""
        self.add_log("Bắt đầu căn chỉnh khuôn mặt...", "info")
        self.status_label.config(text="Đang căn chỉnh khuôn mặt...")
        def align_and_release():
            align_faces(self.add_log)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.camera_active = False
            self.root.after(100, self.update_camera_preview)
        thread = threading.Thread(target=align_and_release)
        thread.daemon = True
        thread.start()

    def run_embed(self):
        """Run face embedding process"""
        self.add_log("Bắt đầu tạo embedding...", "info")
        self.status_label.config(text="Đang tạo embedding...")
        def embed_and_release():
            embed_faces(self.add_log)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.camera_active = False
            self.root.after(100, self.update_camera_preview)
        thread = threading.Thread(target=embed_and_release)
        thread.daemon = True
        thread.start()

    def toggle_recognition(self):
        """Toggle face recognition"""
        if not self.recognition_active:
            # Always re-initialize camera before recognition
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.camera_active = True
                    self.add_log("Camera đã được kích hoạt", "success")
                else:
                    self.add_log("❌ Không thể mở camera cho nhận dạng!", "error")
                    self.status_label.config(text="Không thể mở camera!")
                    return
            self.recognition_active = True
            self.recognition_stop_event.clear()
            self.recognize_btn.config(text="⏹️ Dừng nhận dạng", bg="#dc2626")
            self.add_log("Bắt đầu nhận dạng khuôn mặt", "success")
            self.status_label.config(text="Đang nhận dạng...")

            # Pause camera preview during recognition
            # (update_camera_preview will skip updates while recognition_active)

            def recognition_loop():
                detector = MTCNN(device='cuda:0' if tf.config.list_physical_devices('GPU') else 'cpu')
                db = {}
                embedding_files = list(EMBEDDED_DIR.glob("*.npz"))
                for npz_path in embedding_files:
                    db_name = npz_path.stem.replace("face_database_", "")
                    data = np.load(npz_path)
                    for name in data.files:
                        person_name = name.split('_frame_')[0]
                        if person_name not in db:
                            db[person_name] = []
                        db[person_name].append(normalize([data[name]])[0])
                for person in db:
                    if len(db[person]) > 1:
                        db[person] = np.mean(db[person], axis=0)
                    else:
                        db[person] = db[person][0]
                TEMPLATE = np.array([
                    [38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]
                ], dtype=np.float32)
                logged_names = load_logged_names(ATTENDANCE_LOG_PATH)
                cap = self.cap
                frame_count = 0
                face_tracks = {}
                next_track_id = 0
                detection_interval = 5
                max_distance_threshold = 100
                max_frames_missing = 10
                smoothing_factor = 0.3
                fps_start_time = cv2.getTickCount()
                fps_frame_count = 0
                display_fps = 0
                while self.recognition_active:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    fps_frame_count += 1
                    if fps_frame_count >= 10:
                        fps_end_time = cv2.getTickCount()
                        time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                        display_fps = fps_frame_count / time_diff
                        fps_start_time = fps_end_time
                        fps_frame_count = 0
                    if frame_count % detection_interval == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            results = detector.detect_faces(rgb)
                            # --- START MODIFICATION ---
                            if results: # Nếu có bất kỳ khuôn mặt nào được tìm thấy
                                # Tìm khuôn mặt có diện tích bounding box lớn nhất (w * h)
                                largest_face = max(results, key=lambda face: face['box'][2] * face['box'][3])
                                # Chỉ giữ lại khuôn mặt lớn nhất để xử lý tiếp
                                results = [largest_face]
                            # --- END MODIFICATION ---
                        except:
                            results = []
                        current_detections = []
                        for face in results:
                            x, y, w, h = face['box']
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, frame.shape[1] - x)
                            h = min(h, frame.shape[0] - y)
                            if w <= 0 or h <= 0:
                                continue
                            cx = x + w // 2
                            cy = y + h // 2
                            min_distance = float('inf')
                            matched_track_id = None
                            for track_id, track in face_tracks.items():
                                if track['missing_frames'] > 0:
                                    continue
                                track_cx = track['x'] + track['w'] // 2
                                track_cy = track['y'] + track['h'] // 2
                                distance = np.sqrt((cx - track_cx)**2 + (cy - track_cy)**2)
                                if distance < min_distance and distance < max_distance_threshold:
                                    min_distance = distance
                                    matched_track_id = track_id
                            if matched_track_id is not None:
                                track = face_tracks[matched_track_id]
                                track['x'] = int(track['x'] * (1 - smoothing_factor) + x * smoothing_factor)
                                track['y'] = int(track['y'] * (1 - smoothing_factor) + y * smoothing_factor)
                                track['w'] = int(track['w'] * (1 - smoothing_factor) + w * smoothing_factor)
                                track['h'] = int(track['h'] * (1 - smoothing_factor) + h * smoothing_factor)
                                track['missing_frames'] = 0
                                track['keypoints'] = face['keypoints']
                                track['confidence'] = face.get('confidence', 1.0)
                            else:
                                face_tracks[next_track_id] = {
                                    'x': x, 'y': y, 'w': w, 'h': h,
                                    'missing_frames': 0,
                                    'keypoints': face['keypoints'],
                                    'confidence': face.get('confidence', 1.0),
                                    'name': None,
                                    'similarity': 0.0,
                                    'recognition_count': 0
                                }
                                next_track_id += 1
                            current_detections.append((x, y, w, h))
                        for track_id, track in list(face_tracks.items()):
                            found = False
                            track_cx = track['x'] + track['w'] // 2
                            track_cy = track['y'] + track['h'] // 2
                            for x, y, w, h in current_detections:
                                cx = x + w // 2
                                cy = y + h // 2
                                distance = np.sqrt((cx - track_cx)**2 + (cy - track_cy)**2)
                                if distance < max_distance_threshold:
                                    found = True
                                    break
                            if not found:
                                track['missing_frames'] += 1
                                if track['missing_frames'] > max_frames_missing:
                                    del face_tracks[track_id]
                    for track_id, track in face_tracks.items():
                        if track['missing_frames'] > 0:
                            continue
                        x, y, w, h = track['x'], track['y'], track['w'], track['h']
                        if (frame_count % 15 == 0 or track['name'] is None) and 'keypoints' in track:
                            keypoints = track['keypoints']
                            src = np.array([
                                keypoints['left_eye'], keypoints['right_eye'],
                                keypoints['nose'], keypoints['mouth_left'], keypoints['mouth_right']
                            ], dtype=np.float32)
                            M, _ = cv2.estimateAffinePartial2D(src, TEMPLATE, method=cv2.LMEDS)
                            if M is not None:
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                aligned = cv2.warpAffine(rgb, M, (112, 112))
                                aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
                                temp_path = f"temp_face_{track_id}_{frame_count}.jpg"
                                cv2.imwrite(temp_path, aligned_bgr)
                                try:
                                    result = DeepFace.represent(
                                        img_path=temp_path,
                                        model_name=MODEL_NAME,
                                        detector_backend="skip",
                                        enforce_detection=False
                                    )
                                    if result and isinstance(result, list) and len(result) > 0:
                                        emb = normalize([result[0]["embedding"]])[0]
                                        best_match = None
                                        best_dist = float("inf")
                                        for person_id, stored_emb in db.items():
                                            dist = cosine(emb, stored_emb)
                                            if dist < best_dist:
                                                best_dist = dist
                                                best_match = person_id
                                        similarity = 1 - best_dist
                                        if track['name'] is None or track['recognition_count'] < 5:
                                            if best_dist < THRESHOLD:
                                                track['name'] = best_match
                                                track['similarity'] = similarity
                                                track['recognition_count'] += 1
                                            else:
                                                track['name'] = "Unknown"
                                                track['similarity'] = similarity
                                        else:
                                            track['similarity'] = track['similarity'] * 0.7 + similarity * 0.3
                                except Exception as e:
                                    self.add_log(f"⚠️ Lỗi recognition: {str(e)[:50]}...", "warning")
                                finally:
                                    if os.path.exists(temp_path):
                                        try:
                                            os.remove(temp_path)
                                        except:
                                            pass
                        if track.get('name'):
                            if track['name'] != "Unknown":
                                label = f"{track['name']}"
                                color = (0, 255, 0)
                                if track['name'] not in logged_names and track['recognition_count'] >= 3:
                                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    ATTENDANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                                    with open(ATTENDANCE_LOG_PATH, 'a', newline='') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([track['name'], now, "Present"])
                                    logged_names.add(track['name'])
                                    self.add_log(f"✅ Điểm danh: {track['name']} - {now}", "success")
                            else:
                                label = "Unknown"
                                color = (0, 165, 255)
                        else:
                            label = "Detecting..."
                            color = (255, 255, 0)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
                        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    active_faces = sum(1 for t in face_tracks.values() if t['missing_frames'] == 0)
                    cv2.putText(frame, f"Faces: {active_faces}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Hiển thị lên UI
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.camera_label.config(image=imgtk, text="")
                    self.camera_label.image = imgtk
                    self.root.update_idletasks()
                # cap.release() # Do not release here, let the main control handle it
                for f in Path(".").glob("temp_face_*.jpg"):
                    try:
                        os.remove(f)
                    except:
                        pass
                # Update UI state in main thread
                self.root.after(0, self.stop_recognition_ui_update)

            def stop_recognition_ui_update():
                self.recognition_active = False
                self.recognize_btn.config(text="🔍 Bắt đầu nhận dạng", bg="#ef4444")
                self.status_label.config(text="Ready")
                self.add_log("Đã dừng nhận dạng", "warning")
                # Resume camera preview after recognition
                self.root.after(30, self.update_camera_preview)

            self.recognition_thread = threading.Thread(target=recognition_loop)
            self.recognition_thread.daemon = True
            self.recognition_thread.start()
        else:
            self.recognition_active = False
            self.recognition_stop_event.set()
            # The UI update will be handled by stop_recognition_ui_update once the loop finishes
            
    def export_report(self):
        """Export attendance report"""
        self.add_log("Xuất báo cáo điểm danh", "info")
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename and ATTENDANCE_LOG_PATH.exists():
            import shutil
            shutil.copy(ATTENDANCE_LOG_PATH, filename)
            self.add_log(f"Đã xuất báo cáo: {filename}", "success")
            messagebox.showinfo("Thành công", f"Đã xuất báo cáo tại:\n{filename}")
        else:
            self.add_log("Không có dữ liệu để xuất", "error")
            
    def clear_data(self):
        """Clear all data with confirmation"""
        result = messagebox.askyesno("Xác nhận", 
                                   "Bạn có chắc muốn xóa toàn bộ dữ liệu?\n"
                                   "Hành động này không thể hoàn tác!")
        if result:
            self.add_log("Đang xóa dữ liệu...", "warning")
            try:
                import shutil
                if FRAME_SAVE_ROOT.exists():
                    shutil.rmtree(FRAME_SAVE_ROOT)
                if ALIGN_SAVE_ROOT.exists():
                    shutil.rmtree(ALIGN_SAVE_ROOT)
                if EMBEDDED_DIR.exists():
                    shutil.rmtree(EMBEDDED_DIR)
                if ATTENDANCE_LOG_PATH.exists():
                    os.remove(ATTENDANCE_LOG_PATH)
                    
                self.add_log("Đã xóa dữ liệu thành công!", "success")
                messagebox.showinfo("Thông báo", "Đã xóa dữ liệu thành công!")
                self.update_stats()
            except Exception as e:
                self.add_log(f"Lỗi khi xóa dữ liệu: {str(e)}", "error")
                messagebox.showerror("Lỗi", f"Không thể xóa dữ liệu:\n{str(e)}")
            
    def update_stats(self):
        """Update statistics display"""
        total = 0
        if EMBEDDED_DIR.exists():
            total = len(list(EMBEDDED_DIR.glob("*.npz")))
            self.stats_labels["total_users"].config(text=str(total))
            
        attended = 0
        if ATTENDANCE_LOG_PATH.exists():
            today = datetime.now().strftime("%Y-%m-%d")
            with open(ATTENDANCE_LOG_PATH, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and today in row[1]:
                        attended += 1
            self.stats_labels["attended_today"].config(text=str(attended))
            
            if total > 0:
                rate = (attended / total) * 100
                self.stats_labels["attendance_rate"].config(text=f"{rate:.1f}%")
        
        self.root.after(5000, self.update_stats)
        
    def on_closing(self):
        """Handle window closing"""
        self.recognition_stop_event.set()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


def main_ui():
    """Run UI mode"""
    root = tk.Tk()
    app = AutoAttendUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.update_stats()
    root.mainloop()


def main_cli():
    """Run command line mode"""
    print("🚀 Khởi động AutoAttend System...")
    
    gpu_available = setup_gpu()
    if gpu_available:
        print("⚡ Hệ thống sẽ sử dụng GPU để tăng tốc")
    else:
        print("🐌 Hệ thống sẽ chạy trên CPU")
    
    while True:
        print("\n" + "="*50)
        print("🎯 AutoAttend Menu")
        print("="*50)
        print("1. 📸 Capture Frames (Chụp khung hình)")
        print("2. 🔄 Align Faces (Căn chỉnh khuôn mặt)")
        print("3. 🧠 Embed Faces (Tạo embedding)")
        print("4. 🔍 Recognize Faces (Nhận dạng khuôn mặt)")
        print("5. 🖥️  Launch UI (Giao diện đồ họa)")
        print("6. 🚪 Exit (Thoát)")
        print("="*50)
        
        choice = input("👉 Chọn chức năng (1-6): ").strip()

        if choice == "1":
            name = input("📝 Nhập tên: ").strip()
            user_id = input("🆔 Nhập ID: ").strip()
            if name and user_id:
                capture_frames(name, user_id)
            else:
                print("❌ Vui lòng nhập đầy đủ tên và ID!")
                
        elif choice == "2":
            align_faces()
            
        elif choice == "3":
            embed_faces()
            
        elif choice == "4":
            run_real_time_recognition()
            
        elif choice == "5":
            print("🖥️  Đang khởi động giao diện...")
            main_ui()
            
        elif choice == "6":
            print("👋 Tạm biệt! Cảm ơn bạn đã sử dụng AutoAttend!")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng chọn từ 1-6!")


if __name__ == "__main__":
    # Luôn chạy UI mặc định
    main_ui()