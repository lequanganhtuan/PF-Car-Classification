import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

from config import CFG
# Đảm bảo bạn đã có các hàm này trong inference.py
from inference import load_model_for_inference, load_class_names, get_inference_transform


# ─── Global state ─────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL       = None
CLASS_NAMES = None


def _load_resources():
    """Load model và class names vào global state."""
    global MODEL, CLASS_NAMES

    if MODEL is not None:
        return True, ""

    try:
        # Load class names và ép kiểu thành list string ngay để tránh lỗi replace
        raw_classes = load_class_names(CFG.class_names_path)
        CLASS_NAMES = [v for k, v in sorted(raw_classes.items())]
        
        MODEL = load_model_for_inference(CFG.best_model_path, DEVICE)
        return True, ""
    except Exception as e:
        return False, str(e)


# ─── Prediction function ──────────────────────────────────────────────────────

def predict(image: Image.Image, top_k: int = 5) -> Tuple[Dict, str]:
    ok, err = _load_resources()
    if not ok:
        return {}, f"❌ **Lỗi:** {err}"

    if image is None:
        return {}, "⬆️ Hãy upload một ảnh ô tô để bắt đầu."

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Transform và predict
    transform = get_inference_transform()
    tensor    = transform(image).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        # Sử dụng chuẩn mới torch.amp thay vì torch.cuda.amp
        with torch.amp.autocast('cuda', enabled=CFG.use_amp and DEVICE.type == "cuda"):
            logits = MODEL(tensor)
            probs  = F.softmax(logits, dim=1)[0]
    
    latency = (time.time() - start_time) * 1000

    top_k = min(int(top_k), len(CLASS_NAMES))
    top_probs, top_indices = probs.topk(top_k)

    label_conf = {}
    results    = []

    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        name_raw = CLASS_NAMES[idx]
        display_name = str(name_raw).replace("_", " ").title()

        label_conf[display_name] = round(prob, 4)
        results.append((display_name, round(prob * 100, 1)))

    # Tạo thông tin phản hồi
    top1_name, top1_conf = results[0]
    confidence_level = (
        "🟢 Độ tin cậy cao"   if top1_conf >= 80 else
        "🟡 Độ tin cậy trung bình" if top1_conf >= 50 else
        "🔴 Độ tin cậy thấp (Cần kiểm tra lại)"
    )

    info_md = f"""
### 🏆 Dự đoán: **{top1_name}**
**Xác suất:** {top1_conf:.1f}%  
**Trạng thái:** {confidence_level}
**Thời gian xử lý:** {latency:.0f}ms

---
**Top {len(results)} kết quả:**
"""
    for i, (name, conf) in enumerate(results, 1):
        medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"{i}."
        bar_filled = int(conf / 5)
        bar        = "█" * bar_filled + "░" * (20 - bar_filled)
        info_md   += f"\n{medal} **{name}** — `{bar}` {conf:.1f}%"

    return label_conf, info_md


# ─── UI Layout ────────────────────────────────────────────────────────────────

def create_demo() -> gr.Blocks:
    # Lấy tổng số class
    try:
        names = load_class_names(CFG.class_names_path)
        total_classes = len(names)
    except:
        total_classes = 196

    # Tìm ảnh mẫu
    sample_images = sorted(Path("samples").glob("*.jpg"))[:6] if Path("samples").exists() else []
    sample_images = [[str(p)] for p in sample_images]

    with gr.Blocks(
        title="🚗 Stanford Cars Classifier",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=".title-container { text-align: center; } .subtitle { color: #6b7280; text-align: center; }"
    ) as demo:

        with gr.Column(elem_classes="title-container"):
            gr.Markdown("# 🚗 Stanford Cars Image Classifier")
            gr.Markdown(
                f"Mô hình **ResNet-50** nhận diện **{total_classes} dòng xe ô tô** khác nhau.",
                elem_classes="subtitle",
            )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="📷 Tải ảnh xe lên")
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Hiển thị Top-K")
                predict_btn = gr.Button("🔍 Phân tích ngay", variant="primary")
                clear_btn = gr.Button("🗑️ Xóa")

            with gr.Column(scale=1):
                label_output = gr.Label(label="📊 Xác suất dự đoán", num_top_classes=5)
                info_output = gr.Markdown("⬆️ Upload ảnh và nhấn **Phân tích**.")

        if sample_images:
            gr.Markdown("### 🖼️ Ảnh mẫu để thử")
            gr.Examples(examples=sample_images, inputs=[image_input])

        with gr.Accordion("ℹ️ Thông tin kỹ thuật", open=False):
            info_text = f"""
| Thành phần | Chi tiết |
|-----------|---------|
| **Kiến trúc** | ResNet-50 |
| **Dataset** | Stanford Cars (196 classes) |
| **Thiết bị chạy** | {str(DEVICE).upper()} |
| **Input size** | {CFG.img_size}x{CFG.img_size} |
            """
            gr.Markdown(info_text)

        # Event handlers
        predict_btn.click(fn=predict, inputs=[image_input, top_k_slider], outputs=[label_output, info_output])
        image_input.change(fn=predict, inputs=[image_input, top_k_slider], outputs=[label_output, info_output])
        clear_btn.click(fn=lambda: (None, None, "⬆️ Upload ảnh để bắt đầu."), outputs=[image_input, label_output, info_output])

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Tạo link public")
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(share=args.share)