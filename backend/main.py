from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64

from sop1enhance import kmeans_lab, quantize_image_lab

app = FastAPI()

# Allow React (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_palette(
    image: UploadFile = File(...),
    num_colors: int = Form(...),
):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    pixels = list(img.getdata())

    palette_rgb, centroids_lab, final_mse = kmeans_lab(
        pixels, num_colors
    )

    quantized_img = quantize_image_lab(img, centroids_lab, palette_rgb)

    # Encode quantized image to base64
    buffer = io.BytesIO()
    quantized_img.save(buffer, format="PNG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode()

    palette_hex = [
        "#{:02x}{:02x}{:02x}".format(r, g, b)
        for r, g, b in palette_rgb
    ]

    return {
        "palette": palette_hex,
        "mse": float(round(final_mse, 3)),
        "quantized_image": encoded_img,
    }
