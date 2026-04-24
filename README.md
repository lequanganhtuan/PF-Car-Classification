# 🏎️ Stanford Cars Recognition: End-to-End Deep Learning Pipeline

[![Hugging Face
Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/anhtuan2602/Car_classification)
[![Model](https://img.shields.io/badge/Model-ResNet50-blue)](https://pytorch.org/hub/pytorch_vision_resnet/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)](https://www.python.org/)

This project implements a high-performance image classification system
for **196 car categories** from the **Stanford Cars Dataset**. It
demonstrates a complete AI Engineering workflow, from data handling and
model fine-tuning to Explainable AI (XAI) and cloud deployment.

------------------------------------------------------------------------

## 🧠 Technical Deep Dive

### 1. Model Architecture & Training

-   **Core:** ResNet-50 (Residual Network)
-   **Transfer Learning:** Fine-tuned to output 196 classes
-   **Optimization:** Adam + StepLR, \~12ms inference

### 2. Explainable AI (Grad-CAM)

-   Target Layer: `layer4`
-   Output: Heatmap visualization

### 3. Production Features

-   JSON class mapping
-   Optimized inference with `eval()`
-   Gradio deployment

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── app.py
    ├── model.py
    ├── inference.py
    ├── config.py
    ├── utils/
    │   └── gradcam.py
    ├── outputs/
    │   ├── best_model.pth
    │   └── class_names.json
    └── requirements.txt

------------------------------------------------------------------------

## 🚀 Installation & Usage

``` bash
git clone https://github.com/lequanganhtuan/Week2-project1.git
cd Portfolio2
pip install -r requirements.txt
python app.py
```

------------------------------------------------------------------------

## 🌐 Live Demo

https://huggingface.co/spaces/anhtuan2602/Car_classification
