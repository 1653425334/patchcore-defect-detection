import gradio as gr
import numpy as np
from PIL import Image
from src.anomaly_detector import AnomalyDetector

# 加载模型
detector = AnomalyDetector("memory_bank.pt")

# 判断阈值（后面评估后可以调整）
THRESHOLD = 0.5


def detect(image):
    if image is None:
        return None, None, "请上传图片"

    pil_img = Image.fromarray(image).convert('RGB')
    score, heatmap, overlay = detector.predict(pil_img)

    result = "🔴 异常" if score > THRESHOLD else "🟢 正常"
    label = f"{result}（异常分数：{score:.4f}，阈值：{THRESHOLD}）"

    return heatmap, overlay, label


with gr.Blocks(title="工业缺陷检测系统") as demo:
    gr.Markdown("## 🔍 工业瓶子缺陷检测系统（基于PatchCore）")
    gr.Markdown("上传瓶子图片，系统将自动检测是否存在缺陷并生成异常热力图")

    with gr.Row():
        input_img = gr.Image(label="输入图片")
        heatmap_out = gr.Image(label="异常热力图")
        overlay_out = gr.Image(label="叠加可视化")

    result_label = gr.Textbox(label="检测结果")
    detect_btn = gr.Button("开始检测", variant="primary")

    detect_btn.click(
        fn=detect,
        inputs=input_img,
        outputs=[heatmap_out, overlay_out, result_label]
    )

    gr.Markdown("### 📌 说明\n- 红色区域 = 高异常区域\n- 蓝色区域 = 正常区域\n- 异常分数越高越可能是缺陷")

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

demo.launch(server_name="127.0.0.1", share=False, show_error=True)