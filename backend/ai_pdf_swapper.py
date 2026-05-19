import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import os
import argparse
import io

def get_face_cascade():
    # Load the pre-trained Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

def extract_face(img_path, cascade):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        # Fallback if AI detection fails: just use the center of the image
        h, w = img.shape[:2]
        min_dim = min(h, w)
        cy, cx = h//2, w//2
        r = min_dim // 2
        return img[cy-r:cy+r, cx-r:cx+r], (r*2, r*2)
    
    # Just take the first face found
    x, y, w, h = faces[0]
    # Expand crop slightly to capture the whole head
    padding = int(w * 0.15)
    x1 = max(0, x - padding)
    y1 = max(0, y - int(padding*1.5)) # more padding on top for hair
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    
    face_roi = img[y1:y2, x1:x2]
    return face_roi, (w, h)

def process_pdf(pdf_path, source_face_path, output_path):
    print(f"Loading OpenCV AI Cascade...")
    cascade = get_face_cascade()
    
    print(f"Extracting child's face from {source_face_path}...")
    source_face, original_size = extract_face(source_face_path, cascade)
    
    if source_face is None:
        print("ERROR: Could not read the source face image. Is the path correct?")
        return False

    print(f"Processing Base PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"ERROR: Could not open PDF. {e}")
        return False
        
    output_pdf = fitz.open()

    for page_num in range(len(doc)):
        print(f"  Scanning page {page_num + 1} for characters...")
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
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            print(f"    -> Found character face at coordinates ({x},{y}). Swapping with child's face...")
            
            # Resize source face to fit the detected face area perfectly
            # Add a slight size bump so it covers the original face well
            bump = int(w * 0.1)
            x_adj = max(0, x - bump)
            y_adj = max(0, y - bump)
            w_adj = w + (bump * 2)
            h_adj = h + (bump * 2)
            
            # Ensure it doesn't go out of bounds
            if x_adj + w_adj > img_bgr.shape[1]: w_adj = img_bgr.shape[1] - x_adj
            if y_adj + h_adj > img_bgr.shape[0]: h_adj = img_bgr.shape[0] - y_adj
            
            resized_face = cv2.resize(source_face, (w_adj, h_adj))
            
            # Create a circular feather mask for smooth blending
            mask = np.zeros((h_adj, w_adj), dtype=np.float32)
            cv2.circle(mask, (w_adj//2, h_adj//2), min(w_adj//2, h_adj//2) - 2, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            
            # Extract Region of Interest from the PDF
            roi = img_bgr[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj]
            
            # Blend the original PDF background with the new face using the alpha mask
            roi_float = roi.astype(float)
            face_float = resized_face.astype(float)
            blended = (face_float * mask_3d) + (roi_float * (1.0 - mask_3d))
            
            # Apply back to the image
            img_bgr[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj] = blended.astype(np.uint8)

        # Convert back to PDF page
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Save PIL to bytes
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format="JPEG", quality=90)
        
        # Insert as new PDF page
        img_doc = fitz.open("pdf", fitz.open(stream=img_bytes.getvalue(), filetype="jpeg").convert_to_pdf())
        output_pdf.insert_pdf(img_doc)

    output_pdf.save(output_path)
    output_pdf.close()
    doc.close()
    print(f"\n[SUCCESS] Saved fully AI Face-Swapped PDF to: {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Face Swapper for NexEra PDFs')
    parser.add_argument('--pdf', required=True, help='Path to target Base PDF file')
    parser.add_argument('--face', required=True, help='Path to uploaded child face image (jpg/png)')
    parser.add_argument('--out', required=True, help='Path to output custom PDF')
    args = parser.parse_args()
    
    process_pdf(args.pdf, args.face, args.out)
