# Edge AI & TinyML Lab Manual

**CSE 521: Embedded Machine Learning**
**Author:** Omar Tahon (ID: 022025001)

---

## 1. Introduction to Edge AI & TinyML

Edge AI involves running machine learning algorithms locally on hardware devices (the "edge" of the network) rather than relying on centralized cloud servers. This paradigm shift addresses several critical challenges in modern computing:

*   **Latency:** Processing data locally ensures real-time responses, crucial for applications like autonomous driving and robotics.
*   **Bandwidth:** Sending raw data (like high-res video) to the cloud is expensive; edge AI extracts insights locally.
*   **Privacy \& Security:** Sensitive user data does not need to leave the device.
*   **Power Efficiency:** TinyML specifically focuses on deploying models on ultra-low power microcontrollers (MCUs) consuming milliwatts of power.

---

## 2. State-of-the-Art Edge Models

Recent advancements have drastically reduced the size and computational requirements of models without significantly sacrificing accuracy.

### 2.1 Vision Models
*   **YOLOv11 (Nov 2024):** Introduces a transformer-based backbone with Partial Self-Attention (PSA) and NMS-free training. This results in 25-40% lower latency compared to predecessors, achieving 60+ FPS on edge devices.
*   **MobileNetV4:** Developed by Google, it uses Universal Inverted Bottleneck (UIB) blocks. It achieves 87% ImageNet accuracy with merely 3.8ms latency on mobile edge TPUs.

### 2.2 Small Language Models (SLMs)
Large Language Models (LLMs) are too massive for edge deployment. The industry is pivoting towards optimized SLMs:
*   **Microsoft Phi-3:** A 3.8B parameter model optimized for CPU and mobile inference.
*   **TinyLlama:** A compact 1.1B parameter architecture designed for low-resource environments.

---

## 3. Model Compression and Quantization

To fit these models onto constrained hardware, compression techniques are mandatory. **Quantization** maps high-precision floating-point numbers (FP32) to lower-precision representations (INT8, INT4).

*   **Post-Training Quantization (PTQ):** Convert weights/activations to INT8 post-training. Fast, but can degrade accuracy.
*   **Quantization-Aware Training (QAT):** Simulates low-precision during training to compensate for quantization noise, preserving accuracy.
*   **Activation-aware Weight Quantization (AWQ):** Protects salient (critical) weights, maintaining high accuracy even at 4-bit precision.

---

## 4. Experiment 1: Model Quantization

This conceptual experiment simulates quantization on a randomly generated weight matrix. It mirrors the fundamental operations of inference engines.

```python
import numpy as np

def quantize_int8(weights):
    min_val, max_val = np.min(weights), np.max(weights)
    
    # Calculate scale and zero-point for INT8 [-128, 127]
    scale = (max_val - min_val) / 255.0
    zero_point = -128 - np.round(min_val / scale)
    
    # Quantize and clip
    q_weights = np.round(weights / scale + zero_point)
    q_weights = np.clip(q_weights, -128, 127).astype(np.int8)
    
    return q_weights, scale, zero_point

# 1. Generate dummy float32 weights
weights_fp32 = np.random.randn(10, 10).astype(np.float32) * 5

# 2. Quantize
weights_int8, scale, zp = quantize_int8(weights_fp32)

# 3. Analyze Compression Ratio
print(f"Compression Ratio: {weights_fp32.nbytes / weights_int8.nbytes}x") # ~4.0x
```

---

## 5. Experiment 2: End-to-End Edge Vision Deployment

Using the **Texas Instruments `edgeai-tensorlab`** toolchain as an example, deploying an edge AI model involves a structured pipeline:

1.  **Data Collection (Model Composer):** Capture and annotate images.
2.  **Train & Optimize (`edgeai-modelmaker`):** A command-line tool for Bring Your Own Data (BYOD) training. It handles architecture selection and hyperparameter tuning suitable for Edge processing.
3.  **Compile & Quantize (`edgeai-benchmark`):** Performs Post-Training Quantization (converting the model to fixed-point integers) and compiles the execution graph into machine code optimized for TI's neural processing units (NPUs).
4.  **Deployment (Edge SDK Inference):** Deploy artifacts to the physical board using GStreamer pipelines.

---

## 6. Conclusion

Edge AI transforms how we deploy intelligence in the real world. By leveraging highly efficient model architectures (YOLOv11, MobileNetV4) and rigorous quantization strategies, developers can extract real-time insights from sensor data without the latency and cost of cloud computing. Ecosystems like `edgeai-tensorlab` abstract away the hardware compilation complexity, enabling rapid innovation.
