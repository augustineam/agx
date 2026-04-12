# Agnostix: Unified Machine Learning Architecture

## Vision

A unified, framework-agnostic machine learning architecture built to process X-ray and classical vision data. It provides a seamless transition from classical computer vision (OpenCV) and statistical ML (Scikit-Learn, XGBoost) to modern Deep Learning, with a specific focus on edge deployment (ONNX, C++, Google Coral).

## Core Architectural Principles

1. **Framework Agnosticism via Keras 3**
   Write deep learning models once. Swap between PyTorch and TensorFlow backends based on training speed, cost, or hardware requirements without rewriting architecture or training loops.
2. **Unified API via Scikit-Learn**
   Every component—whether an OpenCV classical feature extractor, a LightGBM model, or a Keras 3 neural network—must adhere to the Scikit-Learn `Pipeline` API (`fit()`, `predict()`).

3. **Strict Environment Isolation via `uv`**
   The project is organized into isolated components to prevent dependency hell (e.g., CUDA conflicts between TF and Torch). `uv` path dependencies are used to share core logic while keeping runner environments pristine.

4. **Edge-First Export Strategy**
   Models are ultimately deployed to C++ via ONNX Runtime and TFLite. ONNX serves as the universal intermediate representation, utilizing `timm` for 1-channel, edge-optimized CNN backbones.

## Project Structure (The "Library & Runner" Pattern)

- **`agnostix/core/`**: Framework-agnostic logic. Uses `keras>=3.0` and `scikit-learn`. Contains base models, data pipelines, and universal `keras.ops` custom layers.
- **`agnostix/backends/torch/`**: PyTorch implementation. Uses `torch`, `skorch`, and local `core`. Contains PyTorch-specific wrappers and data loaders.
- **`agnostix/backends/tf/`**: TensorFlow implementation. Uses `tensorflow`, `scikeras`, and local `core`. Contains TF-specific wrappers and `tf.data` logic.
- **`notebooks/`**: The "working" environments. Houses isolated Jupyter environments with `uv` virtualenvs pointing to the respective backends, keeping the SageMaker deployment payload clean and tiny.
