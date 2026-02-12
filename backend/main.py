from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64

from sop1enhance import kmeans_lab, quantize_image_lab, resize_image, rgb_cube_kmeans_palette

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
    method: str = Form("kmeans"),
):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = resize_image(img)
    pixels = list(img.getdata())

    if method == "median_cut":
        # Use Fast K-Means Original (Huang 2021)
        palette_rgb, final_mse = rgb_cube_kmeans_palette(img, num_colors)
        quantized_img = img.copy()
        arr = list(quantized_img.getdata())
    
        def nearest(c):
            return min(palette_rgb, key=lambda p: (c[0]-p[0])**2 + (c[1]-p[1])**2 + (c[2]-p[2])**2)
        quantized_pixels = [nearest(c) for c in arr]
        quantized_img.putdata(quantized_pixels)
    else:
        # Default: Fast K-Means Enhanced
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
