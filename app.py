# app.py
import os, glob, hashlib, requests, re, time, tempfile, io, json, subprocess, base64
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import streamlit as st
from imageio_ffmpeg import get_ffmpeg_exe  # ⬅️ dùng ffmpeg H.264 để trình duyệt phát được

# ================= Page Config & THEME =================
st.set_page_config(page_title="Realtime Classifier", page_icon="🎞️", layout="wide")

ACCENTS = {
    "Indigo": "#6366f1",
    "Teal":   "#14b8a6",
    "Rose":   "#f43f5e",
    "Amber":  "#f59e0b",
}
accent_pick = st.sidebar.selectbox("🎨 Accent", list(ACCENTS.keys()), index=0)

CUSTOM_CSS = f"""
<style>
:root {{
  --acc: {ACCENTS[accent_pick]};
  --card-bg: rgba(255,255,255,0.6);
  --card-bd: rgba(0,0,0,0.08);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --card-bg: rgba(30,30,30,0.5);
    --card-bd: rgba(255,255,255,0.08);
  }}
}}
.hero {{
  padding: 20px 24px; border-radius: 16px;
  background: linear-gradient(135deg, var(--acc) 0%, #1e293b 60%);
  color: #fff; position: relative; overflow: hidden;
}}
.hero h1 {{ margin: 0; font-weight: 800; letter-spacing: .4px; }}
.hero p  {{ margin: 6px 0 0; opacity: .9; }}

.badge {{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; }}
.badge-acc {{ background: rgba(255,255,255,.18); color:#fff; border:1px solid rgba(255,255,255,.25); }}
.badge-grey{{ background:#F1F3F4; color:#3C4043; }}

.card {{
  background: var(--card-bg);
  border: 1px solid var(--card-bd);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-radius: 14px; padding: 14px 16px;
}}
.info-kv {{
  display: grid; grid-template-columns: max-content 1fr;
  gap: 6px 12px; align-items: center; margin-bottom: 6px;
}}
.info-kv .key {{ color:#64748b; }}
.info-kv .val {{ overflow-wrap: anywhere; word-break: break-word; white-space: normal; }}

hr {{ margin: .6rem 0 1rem; }}
.footer {{ color:#6b7280; font-size:12px; margin-top:24px; text-align:center; }}
.stButton>button {{ border-radius: 12px; }}
.small-hint {{ color:#64748b; font-size:12px; margin-top:-6px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ================= Sidebar =================
st.sidebar.header("⚙️ Cấu hình")

cc1, cc2 = st.sidebar.columns(2)
if cc1.button("🧹 Clear cache"):
    st.cache_resource.clear()
    st.toast("Đã xoá cache. App sẽ tải lại model khi cần.", icon="✅")
if cc2.button("🔁 Rerun"):
    st.experimental_rerun()

dev_choice = st.sidebar.selectbox("Thiết bị chạy", ["Auto (CUDA nếu có)", "GPU (CUDA)", "CPU"], index=0)
cuda_available = torch.cuda.is_available()
if dev_choice == "GPU (CUDA)":
    if not cuda_available:
        st.sidebar.warning("Không phát hiện GPU CUDA. Sẽ dùng CPU.")
    device_kind = "cuda" if cuda_available else "cpu"
elif dev_choice == "CPU":
    device_kind = "cpu"
else:
    device_kind = "cuda" if cuda_available else "cpu"

col_q, col_c = st.sidebar.columns(2)
with col_q:
    use_cpu_quant = st.checkbox("Quantize CPU", value=(device_kind=="cpu"),
                                help="Dynamic quantization (Linear→int8) để tăng tốc trên CPU.")
with col_c:
    use_compile = st.checkbox("torch.compile", value=False,
                              help="Chỉ bật khi môi trường hỗ trợ (Windows cần MSVC; Cloud không có GPU).")

conf_threshold = st.sidebar.slider("Ngưỡng 'không chắc' (confidence)", 0.0, 0.99, 0.20, 0.01)
topk = st.sidebar.slider("Top-K hiển thị", 1, 10, 5)
overlay_pos = st.sidebar.selectbox("Vị trí nhãn trong video", ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"], index=0)
st.sidebar.write(f"Thiết bị: **{device_kind.upper()}**")

with st.sidebar.expander("ℹ️ Hướng dẫn nhanh"):
    st.markdown(
        """
- **B1. Chọn thiết bị:** Auto/GPU/CPU *(CPU có thể bật **Quantize**)*.
- **B2. Ảnh:** kéo-thả **nhiều ảnh** (JPG/PNG) → xem **Top-K** & **confidence** ngay.
- **B3. Video:** upload **MP4/MOV/AVI/MKV** (≤200MB), chọn **FPS sampling** → hệ thống **overlay** nhãn và **phát trực tiếp** trên web.
- **Tuỳ chỉnh:** **threshold**, **Top-K**, vị trí nhãn (Top/Bottom-Left/Right).
"""
    )

# ================== DOWNLOAD WEIGHTS (GitHub Releases) ==================
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def _download_url(url: str, dst: str, headers: dict | None = None):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    base_headers = {"Accept": "application/octet-stream"}
    if headers:
        base_headers.update(headers)
    with requests.get(url, stream=True, timeout=300, headers=base_headers) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1<<20):
                if chunk:
                    f.write(chunk)
    return dst

@st.cache_resource(show_spinner=False)
def ensure_weights() -> str | None:
    # 1) local
    found = glob.glob("models/*.pth") + glob.glob("*.pth")
    if found:
        return found[0]
    # 2) từ URL
    URL = st.secrets.get("WEIGHTS_URL")
    if URL:
        dst = os.path.join("models", os.path.basename(URL))
        token = st.secrets.get("GITHUB_TOKEN")
        headers = {"Authorization": f"token {token}"} if token else None
        _download_url(URL, dst, headers=headers)
        exp = st.secrets.get("WEIGHTS_SHA256")
        if exp:
            got = _sha256(dst)
            if got.lower() != exp.lower():
                try: os.remove(dst)
                except: pass
                raise ValueError(f"SHA256 không khớp. expected={exp} got={got}")
        return dst
    return None

with st.spinner("Đang chuẩn bị trọng số…"):
    ckpt_path = ensure_weights()

if not ckpt_path or not os.path.exists(ckpt_path):
    st.error(
        "❌ Chưa có trọng số.\n\n"
        "Vào **⋯ → Settings → Secrets** và thêm, ví dụ (public):\n"
        "```\nWEIGHTS_URL = \"https://github.com/<user>/<repo>/releases/download/v1.0.0/best_swinB384.pth\"\n"
        "WEIGHTS_SHA256 = \"<checksum-tuỳ-chọn>\"\n```\n"
        "Nếu private, thêm `GITHUB_TOKEN` (PAT)."
    )
    st.stop()

st.sidebar.markdown("**Checkpoint (.pth) (auto)**")
st.sidebar.code(ckpt_path, language="text")

# ================= Labels (fallback) =================
LABELS = []
if os.path.exists("labels_meta.json"):
    try:
        with open("labels_meta.json", "r", encoding="utf-8") as f:
            LABELS = json.load(f).get("class_names", [])
    except Exception as e:
        st.sidebar.warning(f"Không đọc được labels_meta.json: {e}")

# ================= Utils =================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _guess_num_classes(state_dict, fallback=10):
    for k in ["head.weight", "head.fc.weight", "classifier.weight", "fc.weight"]:
        if k in state_dict:
            return state_dict[k].shape[0]
    return fallback

def _optimize_cpu_runtime():
    try:
        n = os.cpu_count() or 4
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
    except Exception:
        pass

def pretty_arch(arch: str) -> str:
    m = re.match(r"swin_(tiny|small|base|large)_patch(\d+)_window(\d+)_?(\d+)?", arch or "")
    if m:
        scale, patch, win, size = m.groups()
        S = {"tiny":"T","small":"S","base":"B","large":"L"}[scale]
        tail = f" • {size}" if size else ""
        s = f"Swin-{S} • p{patch} • win{win}{tail}"
    else:
        s = (arch or "").replace("_", " ")
    return s.replace("•", "•\u200b").replace("-", "-\u200b").replace("/", "/\u200b")

@st.cache_resource(show_spinner=True)
def load_torch_model(ckpt_path: str, device_kind: str, use_cpu_quant: bool, use_compile: bool, labels: list):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    if device_kind == "cpu":
        _optimize_cpu_runtime()
    device = torch.device(device_kind)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    arch = args.get("model", "swin_base_patch4_window12_384")
    img_size = int(args.get("img_size", 384))

    classes_from_ckpt = ckpt.get("classes") if isinstance(ckpt, dict) else None
    class_names = classes_from_ckpt if isinstance(classes_from_ckpt, list) and len(classes_from_ckpt) > 0 else labels

    num_classes = len(class_names) if len(class_names) > 0 else _guess_num_classes(state_dict, fallback=10)

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)

    applied_quant = False
    if device_kind == "cpu" and use_cpu_quant:
        try:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            applied_quant = True
        except Exception as e:
            print("Quantization failed:", e)

    model.eval().to(device)

    applied_compile = False
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, dynamic=False)
            applied_compile = True
        except Exception as e:
            print(f"torch.compile failed: {e}")

    param_count = sum(p.numel() for p in model.parameters())

    return {
        "model": model, "class_names": class_names, "img_size": img_size, "device": device,
        "arch": arch, "applied_quant": applied_quant, "applied_compile": applied_compile,
        "param_count": param_count, "num_classes": num_classes
    }

# Load model
try:
    pack = load_torch_model(ckpt_path, device_kind, use_cpu_quant, use_compile, LABELS)
except Exception as e:
    st.error(f"❌ Lỗi khi load model: {e}")
    st.stop()

model = pack["model"]
CLASS_NAMES = pack["class_names"] if pack["class_names"] else [f"class_{i}" for i in range(pack["num_classes"])]
IMG_SIZE = pack["img_size"]
DEVICE = pack["device"]

# ================= Hero =================
st.markdown(
    f"""
<div class="hero">
  <div style="display:flex; align-items:center; gap:14px;">
    <div style="font-size:28px;">🎬</div>
    <div>
      <h1>Realtime Classifier</h1>
      <p>Ảnh & Video • Thiết kế thân thiện • Hỗ trợ GPU/CPU (quantize) • Top-K & biểu đồ</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ================= Model Info =================
st.subheader("🧠 Thông tin mô hình")
c1, c2, c3, c4, c5 = st.columns([2,2,2,2,2])
with c1:
    st.markdown(f"""
    <div class="card">
      <div class="info-kv"><span class="key">Kiến trúc</span><span class="val"><b>{pretty_arch(pack["arch"])}</b></span></div>
      <div class="info-kv"><span class="key">Ảnh vào</span><span class="val">{IMG_SIZE}×{IMG_SIZE}</span></div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="card">
      <div class="info-kv"><span class="key">Thiết bị</span><span class="val">{DEVICE.type.upper()}</span></div>
      <div class="info-kv"><span class="key">Quantize</span><span class="val">{'ON' if pack['applied_quant'] else 'OFF'}</span></div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="card">
      <div class="info-kv"><span class="key">torch.compile</span><span class="val">{'ON' if pack['applied_compile'] else 'OFF'}</span></div>
      <div class="info-kv"><span class="key">#Classes</span><span class="val">{pack['num_classes']}</span></div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="card">
      <div class="info-kv"><span class="key">#Params</span><span class="val">{pack['param_count']:,}</span></div>
      <div class="info-kv"><span class="key">Labels</span><span class="val">{'OK' if pack['class_names'] else 'Fallback'}</span></div>
    </div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="card"><span class="badge badge-acc">Tip</span> Giữ kích thước ảnh thống nhất giữa train & infer.</div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ================= Preprocess & Inference =================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_pil(pil_img, size=IMG_SIZE):
    arr = np.asarray(pil_img.resize((size, size))).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)

@torch.inference_mode()
def infer_tensor(x: torch.Tensor):
    x = x.to(DEVICE, non_blocking=True)
    t0 = time.perf_counter()
    if DEVICE.type == "cuda":
        with torch.cuda.amp.autocast():
            logits = model(x)
    else:
        logits = model(x)
    dt = time.perf_counter() - t0
    probs = F.softmax(logits[0], dim=-1).detach().float().cpu().numpy()
    return probs, dt

def topk_table(probs: np.ndarray, k: int):
    idxs = np.argsort(-probs)[:k]
    names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}" for i in idxs]
    confs = probs[idxs]
    return pd.DataFrame({"class": names, "confidence": confs}), idxs, confs

def draw_label(np_img_bgr, text, pos="Top-Left"):
    H, W = np_img_bgr.shape[:2]
    h = 42
    rects = {
        "Top-Left":     (0, 0, W, h),
        "Top-Right":    (0, 0, W, h),
        "Bottom-Left":  (0, H-h, W, H),
        "Bottom-Right": (0, H-h, W, H),
    }
    x0, y0, x1, y1 = rects.get(pos, (0,0,W,h))
    cv2.rectangle(np_img_bgr, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(np_img_bgr, text, (12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return np_img_bgr

# ================= Tabs =================
tab_img, tab_vid = st.tabs(["📷 Ảnh", "📼 Video"])

# ---- ẢNH ----
with tab_img:
    st.subheader("Ảnh")
    files = st.file_uploader("Tải 1 hoặc nhiều ảnh (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    st.markdown("<div class='small-hint'>Gợi ý: chọn nhiều ảnh để so sánh kết quả.</div>", unsafe_allow_html=True)

    if files:
        grid = st.columns(2)
        for i, up in enumerate(files):
            try:
                img = Image.open(up).convert("RGB")
            except Exception as e:
                st.error(f"Không đọc được ảnh {up.name}: {e}")
                continue

            probs, dt = infer_tensor(preprocess_pil(img))
            df_top, idxs, confs = topk_table(probs, topk)

            top1_label = CLASS_NAMES[idxs[0]] if idxs[0] < len(CLASS_NAMES) else f"class_{idxs[0]}"
            top1_conf = float(confs[0])
            disp_label = top1_label if top1_conf >= conf_threshold else "Không chắc"

            with grid[i % 2]:
                st.image(img, caption=f"{up.name}", use_container_width=True)
                st.markdown(
                    f"<div class='card'>"
                    f"<b>Kết quả:</b> {disp_label} "
                    f"<span class='badge badge-acc'>Top-1: {top1_conf:.2f}</span> "
                    f"<span class='badge badge-grey'>Time: {dt*1000:.1f} ms</span>"
                    f"</div>", unsafe_allow_html=True)
                st.dataframe(df_top.style.format({"confidence":"{:.3f}"}), use_container_width=True, hide_index=True)
                st.bar_chart(df_top.set_index("class"))

# ---- VIDEO ----
with tab_vid:
    st.subheader("Video")
    left, right = st.columns([1,1])
    with left:
        video = st.file_uploader("Tải video (mp4/mov)", type=["mp4","mov","m4v","avi","mkv"])
        st.markdown("<div class='small-hint'>Limit 200MB • MP4/MOV/AVI/MKV</div>", unsafe_allow_html=True)
    with right:
        default_fps = 3 if DEVICE.type == "cuda" else 2
        fps_proc = st.slider("FPS suy luận (sampling)", 1, 30, default_fps)
        autoplay = st.checkbox("Tự phát (autoplay, muted)", value=False)

    if video:
        with st.status("Đang xử lý video…", expanded=False) as status:
            tdir = tempfile.mkdtemp()
            in_path  = os.path.join(tdir, "in.mp4")
            out_path = os.path.join(tdir, "out.mp4")
            with open(in_path, "wb") as f:
                f.write(video.read())

            cap = cv2.VideoCapture(in_path)
            if not cap.isOpened():
                st.error("Không mở được video.")
            else:
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vfps   = cap.get(cv2.CAP_PROP_FPS)
                total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
                fps_out = vfps if vfps and vfps > 0 else 25

                # === ffmpeg H.264 (avc1) + yuv420p + faststart ===
                try:
                    ffmpeg = get_ffmpeg_exe()
                except Exception as e:
                    st.error(f"Không tìm thấy ffmpeg: {e}\nHãy thêm `imageio-ffmpeg` vào requirements.")
                    st.stop()

                cmd = [
                    ffmpeg, "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}",
                    "-r", str(fps_out),
                    "-i", "-",
                    "-an",
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-movflags", "+faststart",
                    out_path,
                ]

                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                except Exception as e:
                    st.error(f"Không khởi chạy được ffmpeg/libx264: {e}")
                    st.stop()

                pbar = st.progress(0)
                frame_id, infered, t_infer, last_infer_ts = 0, 0, 0.0, -1.0
                infer_interval = 1.0 / max(1, fps_proc)
                last_text = ""

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    t = frame_id / fps_out

                    if last_infer_ts < 0 or (t - last_infer_ts) >= infer_interval:
                        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        probs, dt = infer_tensor(preprocess_pil(pil))
                        idx = int(np.argmax(probs))
                        conf = float(probs[idx])
                        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
                        disp_label = label if conf >= conf_threshold else "Unlikely"
                        last_text = f"{disp_label} (p={conf:.2f})"
                        last_infer_ts = t
                        infered += 1
                        t_infer += dt

                    # overlay và gửi frame vào ffmpeg encoder
                    frame = draw_label(frame, last_text, pos=overlay_pos)
                    try:
                        proc.stdin.write(frame.tobytes())
                    except Exception as e:
                        st.error(f"ffmpeg ghi lỗi: {e}")
                        break

                    frame_id += 1
                    if total: pbar.progress(min(frame_id / total, 1.0))

                cap.release()
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                ret = proc.wait()
                if ret != 0:
                    st.error("ffmpeg trả về mã lỗi (có thể thiếu libx264).")
                    st.stop()

                status.update(label="Hoàn tất ✅", state="complete")
                st.toast("Xong! Video đã được gắn nhãn (H.264).", icon="🎉")

                # Phát trực tiếp trên web
                try:
                    with open(out_path, "rb") as f:
                        video_bytes = f.read()
                except Exception as e:
                    st.error(f"Không đọc được video output: {e}")
                    st.stop()

                if autoplay:
                    b64 = base64.b64encode(video_bytes).decode("utf-8")
                    st.markdown(
                        f"""
                        <video controls autoplay muted playsinline style="width:100%; border-radius:12px;">
                          <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                          Trình duyệt không hỗ trợ phát video.
                        </video>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.video(video_bytes, format="video/mp4")

                # Nút tải (tuỳ chọn)
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Tải video đã gắn nhãn", f, file_name="result.mp4", mime="video/mp4")

                avg_ms = (t_infer / max(infered,1)) * 1000.0
                est_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
                st.markdown(
                    f"<div class='card'>"
                    f"<span class='badge badge-acc'>Avg Infer: {avg_ms:.1f} ms</span> "
                    f"<span class='badge badge-acc'>≈ {est_fps:.1f} FPS</span> "
                    f"<span class='badge badge-grey'>Frames: {frame_id}</span>"
                    f"</div>", unsafe_allow_html=True
                )

# ================= Footer =================
st.markdown("<div class='footer'>Made with ❤️ SMC_4</div>", unsafe_allow_html=True)
