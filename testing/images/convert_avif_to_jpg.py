from PIL import Image
import os

BASE_DIR = os.path.dirname(__file__)
src = BASE_DIR
dst = os.path.join(BASE_DIR, "jpg")

os.makedirs(dst, exist_ok=True)

for f in os.listdir(src):
    if f.lower().endswith(".avif"):
        img = Image.open(os.path.join(src, f))
        out = os.path.splitext(f)[0] + ".jpg"
        img.convert("RGB").save(os.path.join(dst, out), "JPEG", quality=95)
        print("Converted:", f)

python src/facesense_cli.py testing/images/angry.png
python src/facesense_cli.py data/raw/angry2.jpg

