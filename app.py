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
WEIGHTS = os.getenv("MODEL_WEIGHTS", "best_unet.pt")

if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(
        f"Model weights '{WEIGHTS}' not found.\n"
        f"Please download best_unet.pt from your Hugging Face model repository/Space "
        f"and place it in the project root directory next to app.py."
    )

import segmentation_models_pytorch as smp

model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.to(device).eval()
print(f"Loaded model weights from '{WEIGHTS}' onto {device}.")

def segment(path):
    """
    Inference pipeline:
    1. Reads 3-channel MRI slice (.tif or .png).
    2. Performs per-patient z-score normalization on brain tissue.
    3. Predicts tumor mask with U-Net.
    4. Overlays outer boundary in red.
    """
    if path is None:
        return None, "Please upload a valid MRI slice."

    raw = imread(path)
    if raw.ndim == 2:
        raw = np.stack([raw] * 3, -1)
    elif raw.shape[-1] > 3:
        raw = raw[..., :3]

    img = raw[..., :3].astype(np.float32)
    flair = img[..., 1].copy()  # FLAIR channel shown to the user

    # Per-channel normalization matching training distribution
    for c in range(3):
        ch = img[..., c]
        brain = ch[ch > 0]
        if brain.size:
            img[..., c] = np.where(ch > 0, (ch - brain.mean()) / (brain.std() + 1e-8), 0)

    with torch.no_grad():
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        mask = (torch.sigmoid(model(x)) > 0.5)[0, 0].cpu().numpy()

    flair_ptp = np.ptp(flair) + 1e-8
    base = (255 * (flair - flair.min()) / flair_ptp).astype(np.uint8)
    out = np.stack([base] * 3, -1)
    out[find_boundaries(mask, mode="outer")] = [255, 0, 0]  # Red outline

    area = int(mask.sum())
    result_text = f"Tumor area: {area:,} pixels" if area else "No tumor detected"
    return out, result_text

# Load sample slices if present
samples_dir = "samples"
examples = None
if os.path.isdir(samples_dir):
    valid_exts = (".tif", ".png", ".jpg", ".jpeg")
    sample_files = [
        os.path.join(samples_dir, f) for f in sorted(os.listdir(samples_dir))
        if f.lower().endswith(valid_exts) and not f.endswith("_mask.tif")
    ]
    if sample_files:
        examples = [[f] for f in sample_files]

demo = gr.Interface(
    fn=segment,
    inputs=gr.Image(label="Upload an MRI slice (3-channel .tif/.png)", type="filepath"),
    outputs=[
        gr.Image(label="Tumor outline (Red)"),
        gr.Textbox(label="Result"),
    ],
    title="🧠 Brain Tumor Segmentation (U-Net)",
    description="Research demo on the LGG MRI dataset using U-Net (ResNet-34). Not a medical device.",
    article="**Note**: Requires `best_unet.pt` placed in the project root directory.",
    examples=examples,
)

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(server_name=server_name, server_port=7860)
