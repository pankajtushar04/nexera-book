"""
NexEra Smart Face Blender
--------------------------
Cartoon-aware face personalization using:
  1. InsightFace landmark detection (5-point keypoints)
  2. Affine warp to align real face geometry to cartoon face geometry
  3. LAB color histogram matching to match cartoon color palette
  4. Elliptical feathered mask for clean edges
  5. Poisson seamless cloning (MIXED mode) for invisible edges
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import os
import io
from insightface.app import FaceAnalysis

print("\n--- Initializing Smart Cartoon Blender ---")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

app_small = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_small.prepare(ctx_id=-1, det_size=(320, 320))
print("--- Smart Blender Ready ---\n")


def detect_face_robust(img):
    """Detect face using multiple strategies for reliability."""
    faces = app.get(img)
    if faces:
        return sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)

    # Retry with smaller detector
    faces = app_small.get(img)
    if faces:
        return sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)

    # Retry with 2× upscale
    big = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    faces = app.get(big)
    if faces:
        # Scale keypoints back down
        for f in faces:
            f.kps = f.kps / 2.0
            f.bbox = f.bbox / 2.0
        return sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)

    return []


def match_color_lab(src_face, dst_face_region):
    """
    Transfer color characteristics from dst_face_region (cartoon) to src_face (real).
    Works in LAB color space to preserve luminance structure while matching cartoon palette.
    """
    src_lab = cv2.cvtColor(src_face, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_face_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        src_mean, src_std = src_lab[:,:,ch].mean(), src_lab[:,:,ch].std() + 1e-6
        dst_mean, dst_std = dst_lab[:,:,ch].mean(), dst_lab[:,:,ch].std() + 1e-6
        # Scale and shift the channel to match the cartoon's distribution
        src_lab[:,:,ch] = (src_lab[:,:,ch] - src_mean) * (dst_std / src_std) + dst_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def cartoonify_face(face_img):
    """
    Apply a subtle cartoon stylization to the face to blend with illustrated backgrounds.
    - Bilateral filter smooths skin while preserving edges (key cartoon look)
    - Slight edge enhancement
    """
    # Multiple bilateral filter passes for strong smoothing
    smooth = face_img.copy()
    for _ in range(5):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

    # Detect edges for cartoon lines
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 5), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine smoothed face with edge overlay
    cartoon = cv2.bitwise_and(smooth, edges_bgr)
    # Blend original with cartoon 40/60 to keep some realism
    return cv2.addWeighted(face_img, 0.4, cartoon, 0.6, 0)


def warp_face_to_target(src_img, src_kps, dst_shape, dst_kps):
    """
    Warp source face image so its keypoints align with the destination cartoon face keypoints.
    Uses partial affine (rotation + scale + translation) for natural-looking alignment.
    """
    M, _ = cv2.estimateAffinePartial2D(src_kps, dst_kps, method=cv2.RANSAC)
    if M is None:
        # Fall back to simple translation if affine fails
        dx = dst_kps[2][0] - src_kps[2][0]  # nose tip offset
        dy = dst_kps[2][1] - src_kps[2][1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])

    h, w = dst_shape[:2]
    warped = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped, M


def create_face_mask(dst_kps, img_shape, scale=1.35):
    """
    Create a smooth elliptical mask centered on the cartoon face,
    sized to cover the face region with feathered edges.
    """
    h, w = img_shape[:2]

    # Compute face center from keypoints (left eye, right eye, nose, left mouth, right mouth)
    center_x = int(dst_kps[:, 0].mean())
    center_y = int(dst_kps[:, 1].mean())

    # Face size from eye distance
    eye_dist = np.linalg.norm(dst_kps[0] - dst_kps[1])
    radius_x = int(eye_dist * scale * 1.0)
    radius_y = int(eye_dist * scale * 1.4)

    # Shift mask slightly upward to cover forehead
    center_y = max(radius_y, center_y - int(eye_dist * 0.2))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)

    # Heavy feathering for invisible edges
    feather = max(15, int(eye_dist * 0.25))
    mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), feather)
    return mask


def blend_face_onto_cartoon(cartoon_page, src_face_img, src_kps, cartoon_face):
    """
    Full pipeline: warp → cartoonify → color-match → seamless clone onto cartoon page.
    """
    dst_kps = cartoon_face.kps
    dst_bbox = cartoon_face.bbox.astype(int)

    # 1. Warp real face to match cartoon face geometry
    warped, M = warp_face_to_target(src_face_img, src_kps, cartoon_page.shape, dst_kps)

    # 2. Cartoonify the warped face
    cartoonified = cartoonify_face(warped)

    # 3. Extract the cartoon face region for color reference
    x1, y1, x2, y2 = max(0,dst_bbox[0]), max(0,dst_bbox[1]), min(cartoon_page.shape[1],dst_bbox[2]), min(cartoon_page.shape[0],dst_bbox[3])
    if x2 > x1 and y2 > y1:
        cartoon_face_region = cartoon_page[y1:y2, x1:x2]
        # Get the same region from our warped face
        warped_region = cartoonified[y1:y2, x1:x2]
        if warped_region.shape == cartoon_face_region.shape and warped_region.size > 0:
            # Apply color transfer only on the face region, then put back
            color_matched_region = match_color_lab(warped_region, cartoon_face_region)
            cartoonified[y1:y2, x1:x2] = color_matched_region

    # 4. Create feathered elliptical face mask
    mask = create_face_mask(dst_kps, cartoon_page.shape)

    # 5. Compute center for seamlessClone
    center_x = int(dst_kps[:, 0].mean())
    center_y = int(dst_kps[:, 1].mean())
    center = (
        np.clip(center_x, 1, cartoon_page.shape[1]-2),
        np.clip(center_y, 1, cartoon_page.shape[0]-2)
    )

    # 6. Poisson seamless clone - MIXED_CLONE preserves the cartoon texture/style
    try:
        result = cv2.seamlessClone(
            cartoonified, cartoon_page, mask, center, cv2.MIXED_CLONE
        )
    except Exception as e:
        print(f"    Warning: seamlessClone failed ({e}), using alpha blend fallback")
        alpha = mask.astype(np.float32)[:,:,None] / 255.0
        result = (cartoonified.astype(np.float32) * alpha + cartoon_page.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    return result


def process_pdf_smart(pdf_path, source_face_path, output_path):
    """Main entry point: process every page of the PDF with smart cartoon face blending."""
    print(f"\n[Smart Blender] Loading source face from {source_face_path}...")

    # Load source image robustly
    try:
        pil_img = Image.open(source_face_path).convert('RGB')
        src_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        print(f"  Source image: {src_img.shape[1]}x{src_img.shape[0]}")
    except Exception as e:
        print(f"  ERROR loading image: {e}")
        return False

    # Detect face in source image
    src_faces = detect_face_robust(src_img)
    if not src_faces:
        print("  ERROR: No face detected in source photo!")
        return False

    src_face = src_faces[0]
    src_kps = src_face.kps
    print(f"  Source face detected at bbox: {src_face.bbox.astype(int)}")

    # Open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ERROR opening PDF: {e}")
        return False

    output_pdf = fitz.open()
    total_swaps = 0

    for page_num in range(len(doc)):
        print(f"\n  Processing page {page_num + 1}/{len(doc)}...")
        page = doc.load_page(page_num)

        # Render at 2× resolution for quality
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)

        # Convert to numpy BGR
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        page_img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

        # Detect faces on the cartoon page
        page_faces = detect_face_robust(page_img)

        if not page_faces:
            print(f"    No characters found on page {page_num+1}, keeping original.")
        else:
            print(f"    Found {len(page_faces)} character(s) to personalize.")
            result = page_img.copy()
            for i, cartoon_face in enumerate(page_faces):
                print(f"    Blending character {i+1} with smart cartoon-aware technique...")
                result = blend_face_onto_cartoon(result, src_img, src_kps, cartoon_face)
                total_swaps += 1
            page_img = result

        # Convert back to PDF page
        img_rgb = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
        pil_page = Image.fromarray(img_rgb)
        img_bytes = io.BytesIO()
        pil_page.save(img_bytes, format="JPEG", quality=92)
        img_doc = fitz.open("pdf", fitz.open(stream=img_bytes.getvalue(), filetype="jpeg").convert_to_pdf())
        output_pdf.insert_pdf(img_doc)

    output_pdf.save(output_path)
    output_pdf.close()
    doc.close()
    print(f"\n[Smart Blender] Done! Applied {total_swaps} personalized face(s). Saved to: {output_path}")
    return True
