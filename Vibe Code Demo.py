
# 🧪 StyleGen Code Demo (Python script version)
# Demonstrates how to run Stable Diffusion to generate a fashion outfit image from a text prompt

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to(device)

# Fashion prompt
prompt = "a red velvet cocktail dress with gold embroidery"

# Generate image
image = pipe(prompt).images[0]

# Show the image
image.show()
