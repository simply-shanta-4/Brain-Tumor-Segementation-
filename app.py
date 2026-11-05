"""
Brain Tumor Segmentation - Gradio Web Application
------------------------------------------------
Entry point for local demonstration and Hugging Face Spaces.
Uses a U-Net (ResNet-34 backbone) trained on LGG MRI FLAIR sequences
to detect and outline brain tumor boundaries.
"""

import os
import numpy as np
import torch
import gradio as gr
from skimage.io import imread
from skimage.segmentation import find_boundaries

# Determine execution device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration and weight path
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "best_unet.pt")
model = None

def load_segmentation_model():
    """Initializes and loads the U-Net model with weights if available."""
    global model
    try:
        import segmentation_models_pytorch as smp
        net = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
        if os.path.exists(MODEL_WEIGHTS):
            net.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
            print(f"Loaded weights from {MODEL_WEIGHTS} onto {device}")
        else:
            print(f"Notice: Model weight file '{MODEL_WEIGHTS}' not found. Initialized with random weights for demo.")
        net.to(device).eval()
        return net
    except Exception as e:
        print(f"Warning: Could not load segmentation model: {e}")
        return None

model = load_segmentation_model()

def preprocess_slice(raw_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalizes the input MRI slice matching training distribution:
    Per-channel z-score computed over non-zero brain tissue.
    """
    if raw_img.ndim == 2:
        raw_img = np.stack([raw_img] * 3, -1)
    elif raw_img.shape[-1] > 3:
        raw_img = raw_img[..., :3]

    img = raw_img[..., :3].astype(np.float32)
    flair = img[..., 1].copy()  # FLAIR channel for display

    for c in range(3):
        channel = img[..., c]
        brain_pixels = channel[channel > 0]
        if brain_pixels.size > 0:
            mean = brain_pixels.mean()
            std = brain_pixels.std() + 1e-8
            img[..., c] = np.where(channel > 0, (channel - mean) / std, 0.0)

    return img, flair

def segment_mri(input_file):
    """
    Gradio prediction pipeline:
    1. Read input image (support path or numpy array)
    2. Preprocess & normalize MRI slice
    3. Run U-Net inference to predict binary tumor mask
    4. Highlight tumor contour in red
    5. Return marked visualization and quantitative statistics
    """
    if input_file is None:
        return None, "Please upload a valid MRI slice (.tif, .png, or .jpg)."

    try:
        if isinstance(input_file, str):
            raw = imread(input_file)
        else:
            raw = np.array(input_file)

        norm_img, flair = preprocess_slice(raw)

        # Run inference
        if model is not None:
            with torch.no_grad():
                tensor_input = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)
                logits = model(tensor_input)
                pred_prob = torch.sigmoid(logits)
                mask = (pred_prob > 0.5)[0, 0].cpu().numpy().astype(bool)
        else:
            # Fallback heuristic: FLAIR intensity threshold
            brain_mask = flair > 10
            if brain_mask.sum() > 0:
                thresh = flair[brain_mask].mean() + 1.25 * flair[brain_mask].std()
                mask = (flair > thresh) & brain_mask
            else:
                mask = np.zeros_like(flair, dtype=bool)

        # Generate red boundary overlay on normalized grayscale FLAIR
        flair_ptp = np.ptp(flair) + 1e-8
        base_gray = (255 * (flair - flair.min()) / flair_ptp).astype(np.uint8)
        overlay = np.stack([base_gray] * 3, axis=-1)

        # Draw red outline along tumor boundary
        boundaries = find_boundaries(mask, mode="outer")
        overlay[boundaries] = [255, 0, 0]

        tumor_area = int(mask.sum())
        total_pixels = mask.size
        area_pct = (tumor_area / total_pixels) * 100

        if tumor_area > 0:
            result_text = f"Tumor Detected\n• Estimated Area: {tumor_area:,} pixels ({area_pct:.2f}% of slice)\n• Mask boundary outlined in red."
        else:
            result_text = "No Tumor Detected\n• Zero pixels above confidence threshold."

        return overlay, result_text

    except Exception as err:
        return None, f"Error processing MRI slice: {str(err)}"

# Gradio Interface Definition
samples_dir = "samples"
examples = []
if os.path.isdir(samples_dir):
    valid_extensions = (".tif", ".png", ".jpg", ".jpeg")
    sample_files = [
        os.path.join(samples_dir, f) for f in sorted(os.listdir(samples_dir))
        if f.lower().endswith(valid_extensions) and not f.endswith("_mask.tif")
    ]
    if sample_files:
        examples = [[f] for f in sample_files[:6]]

demo = gr.Interface(
    fn=segment_mri,
    inputs=gr.Image(label="Upload MRI Slice (FLAIR / Multimodal)", type="filepath"),
    outputs=[
        gr.Image(label="Segmentation Overlay (Red Outline)", type="numpy"),
        gr.Textbox(label="Analysis & Quantitative Output", lines=3),
    ],
    title="🧠 Brain Tumor Segmentation (U-Net)",
    description=(
        "Automated semantic segmentation of low-grade gliomas (LGG) on brain MRI slices using a deep U-Net architecture. "
        "Upload a 3-channel MRI slice (.tif, .png) to segment tumor tissue with high precision."
    ),
    article="**Disclaimer**: This demo is intended for academic and research evaluation only. It is not an FDA/CE-approved medical device.",
    examples=examples if len(examples) > 0 else None,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
