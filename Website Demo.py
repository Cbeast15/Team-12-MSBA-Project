import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load model on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
).to(device)

def generate_fashion(prompt):
    image = pipe(prompt).images[0]
    return image

demo = gr.Interface(
    fn=generate_fashion,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Describe an outfit (e.g., 'a red velvet gown with gold embroidery')",
        label="Fashion Prompt"
    ),
    outputs="image",
    title="👗 StyleGen: AI Fashion Generator",
    description="Enter a fashion concept and generate a styled image using Stable Diffusion."
)

demo.launch()
