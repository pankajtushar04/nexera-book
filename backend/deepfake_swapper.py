import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import os
import urllib.request
import insightface
from insightface.app import FaceAnalysis
import io

# URLs for required models
SWAPPER_MODEL_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
SWAPPER_MODEL_PATH = "inswapper_128.onnx"

print("\n--- Initializing Deepfake Engine ---")
if not os.path.exists(SWAPPER_MODEL_PATH):
    print("Downloading 530MB Face Swap Model (inswapper_128.onnx)... This is a one-time download and may take a few minutes!")
    urllib.request.urlretrieve(SWAPPER_MODEL_URL, SWAPPER_MODEL_PATH)
    print("Download complete!")

print("Loading Face Analysis Model (Buffalo_L)...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

app_small = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_small.prepare(ctx_id=-1, det_size=(320, 320))

print("Loading Inswapper Model...")
swapper = insightface.model_zoo.get_model(SWAPPER_MODEL_PATH, download=False, download_zip=False)
print("--- Deepfake Engine Ready ---\n")

def process_pdf_deepfake(pdf_path, source_face_path, output_path):
    # Use PIL to read image - handles all formats (JPEG, PNG, WEBP, etc.) from browser
    try:
        pil_img_src = Image.open(source_face_path).convert('RGB')
        source_img = cv2.cvtColor(np.array(pil_img_src), cv2.COLOR_RGB2BGR)
        print(f"  Source image size: {source_img.shape[1]}x{source_img.shape[0]}")
    except Exception as e:
        print(f"ERROR: Could not read source image: {e}")
        return False

    # Try detection at multiple scales - handles low-res, far, or partially visible faces
    source_faces = app.get(source_img)
    print(f"  Detection attempt 1 (640x640): found {len(source_faces)} faces")
    
    if len(source_faces) == 0:
        print("  Retrying with small detector (320x320)...")
        source_faces = app_small.get(source_img)
        print(f"  Detection attempt 2 (320x320): found {len(source_faces)} faces")
    
    if len(source_faces) == 0:
        print("  Retrying with 2x upscaled image...")
        upscaled = cv2.resize(source_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        source_faces = app.get(upscaled)
        print(f"  Detection attempt 3 (upscaled): found {len(source_faces)} faces")
    
    if len(source_faces) == 0:
        print("ERROR: No face detected after 3 attempts.")
        return False
    
    # Sort faces by size and take the largest one (most prominent)
    source_faces = sorted(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    source_face = source_faces[0]
    print(f"  Face found! Using best face for swap.")
    
    print(f"Processing Base PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"ERROR: Could not open PDF. {e}")
        return False
        
    output_pdf = fitz.open()

    for page_num in range(len(doc)):
        print(f"  Scanning and Swapping page {page_num + 1}...")
        page = doc.load_page(page_num)
        
        # Render page to image at high resolution (zoom x2)
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert PyMuPDF pixmap to numpy array for OpenCV
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        # Detect faces on the PDF page
        target_faces = app.get(img_bgr)
        
        result_img = img_bgr.copy()
        
        for target_face in target_faces:
            print(f"    -> Swapping a character's face with seamless deepfake...")
            # This applies the ONNX Inswapper model to generate a seamless, tone-matched face replacement!
            result_img = swapper.get(result_img, target_face, source_face, paste_back=True)

        # Convert back to PDF page
        img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format="JPEG", quality=95)
        
        # Insert as new PDF page
        img_doc = fitz.open("pdf", fitz.open(stream=img_bytes.getvalue(), filetype="jpeg").convert_to_pdf())
        output_pdf.insert_pdf(img_doc)

    output_pdf.save(output_path)
    output_pdf.close()
    doc.close()
    print(f"\n[SUCCESS] Saved flawless Deepfake PDF to: {output_path}")
    return True
