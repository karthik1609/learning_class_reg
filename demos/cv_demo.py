from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def generate_synthetic_image(width: int = 128, height: int = 128) -> Image.Image:
    rng = np.random.default_rng(42)
    # Create three color regions
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, : width // 3] = [255, 0, 0]  # red
    img[:, width // 3 : 2 * width // 3] = [0, 255, 0]  # green
    img[:, 2 * width // 3 :] = [0, 0, 255]  # blue
    # Add some noise
    noise = rng.integers(0, 30, size=img.shape, endpoint=True, dtype=np.uint8)
    img = np.clip(img + noise, 0, 255)
    return Image.fromarray(img, mode="RGB")


def kmeans_color_quantization(image: Image.Image, n_colors: int = 3) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    h, w, c = arr.shape
    pixels = arr.reshape(-1, c)

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    quantized = centers[labels].reshape(h, w, c)
    return Image.fromarray((quantized * 255).astype(np.uint8), mode="RGB")


def main() -> None:
    img = generate_synthetic_image()
    quant = kmeans_color_quantization(img, n_colors=3)
    # Show basic stats rather than opening a window (CLI friendly)
    print(f"Original size: {img.size}, Quantized size: {quant.size}")
    # Save outputs to disk
    img.save("demos_outputs_cv_original.png")
    quant.save("demos_outputs_cv_kmeans.png")
    print("Saved demos_outputs_cv_original.png and demos_outputs_cv_kmeans.png")


if __name__ == "__main__":
    main()


