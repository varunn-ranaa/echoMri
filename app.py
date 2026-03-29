import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model.generator import CycleGenerator
import os

port = int(os.environ.get("PORT", 10000))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_echo2mri = CycleGenerator().to(device)
G_echo2mri.load_state_dict(torch.load(
    'checkpoints/G_echo2mri_ep10.pth',
    map_location=device))
G_echo2mri.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def enhance(image):
    # None check
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = G_echo2mri(img_tensor)
    output = output.squeeze().cpu().numpy()
    output = ((output + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(output)

with gr.Blocks() as demo:
    gr.HTML("""
        <div style='text-align:center; padding: 32px 0 16px'>
            <h1 style='font-size:28px; font-weight:600; color:#1a1a2e; margin:0'>
                Echo → MRI
            </h1>
            <p style='color:#6c757d; margin-top:8px; font-size:15px'>
                Echocardiography image to MRI quality like enhanced
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(
                type="pil",
                label="Upload Echo Image",
                height=300
            )
            btn = gr.Button("✦ Enhance", variant="primary")

        with gr.Column():
            out = gr.Image(
                label="Generated Enhanced Echo",
                height=300
            )

    btn.click(fn=enhance, inputs=inp, outputs=out)

    gr.HTML("""
        <div style='text-align:center; padding:24px 0 8px;
                    color:#adb5bd; font-size:13px'>
            CycleGAN · Unpaired Training · 3583 Echo + MRI images
        </div>
    """)

demo.launch(server_name="0.0.0.0", server_port=port)
