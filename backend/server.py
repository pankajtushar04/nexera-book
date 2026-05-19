from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import shutil
import os
import requests
import time

app = FastAPI()

DATABASE_FILE = os.path.join(os.path.dirname(__file__), "database.json")

class StoryItem(BaseModel):
    id: int
    title: str
    emoji: str
    status: str
    pages: int
    fileData: str = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NEXERA_DIR = os.path.join(os.path.dirname(__file__), "..")
COLAB_URL_FILE = os.path.join(os.path.dirname(__file__), "colab_url.txt")

STATIC_DIR = os.path.join(NEXERA_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class SettingsModel(BaseModel):
    colab_url: str

@app.get("/api/stories")
async def get_stories():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return JSONResponse(data)
            except Exception:
                pass
    # Return default templates if the database file is empty/missing
    return JSONResponse([
      { "id": 1, "title": "Space Explorer", "emoji": "🚀", "status": "Active", "pages": 12 },
      { "id": 2, "title": "Marine Biologist", "emoji": "🐋", "status": "Active", "pages": 10 },
      { "id": 3, "title": "Jungle Explorer", "emoji": "🦖", "status": "Active", "pages": 14 }
    ])

@app.post("/api/stories")
async def save_stories(stories: List[StoryItem]):
    with open(DATABASE_FILE, "w", encoding="utf-8") as f:
        json.dump([story.model_dump() for story in stories], f, ensure_ascii=False, indent=2)
    return JSONResponse({"status": "success"})

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

@app.post("/api/stories/upload")
async def upload_story_template(
    title: str = Form(...),
    emoji: str = Form(...),
    file: UploadFile = File(...)
):
    story_id = int(time.time() * 1000)
    file_ext = os.path.splitext(file.filename)[1] or ".pdf"
    file_name = f"{story_id}{file_ext}"
    file_path = os.path.join(TEMPLATES_DIR, file_name)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    stories = []
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r", encoding="utf-8") as f:
            try:
                stories = json.load(f)
            except Exception:
                pass
                
    new_story = {
        "id": story_id,
        "title": title,
        "emoji": emoji,
        "status": "Active",
        "pages": 12,
        "fileData": f"/api/templates/{file_name}"
    }
    
    stories.insert(0, new_story)
    
    with open(DATABASE_FILE, "w", encoding="utf-8") as f:
        json.dump(stories, f, ensure_ascii=False, indent=2)
        
    return JSONResponse({"status": "success", "story": new_story})

@app.get("/api/templates/{file_name}")
async def get_template_file(file_name: str):
    file_path = os.path.join(TEMPLATES_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf")
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.get("/")
async def serve_frontend():
    html_path = os.path.join(NEXERA_DIR, "nexera-standalone.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/admin")
async def serve_admin():
    html_path = os.path.join(NEXERA_DIR, "nexera-admin.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/api/settings")
async def get_settings():
    colab_url = ""
    if os.path.exists(COLAB_URL_FILE):
        with open(COLAB_URL_FILE, "r", encoding="utf-8") as f:
            colab_url = f.read().strip()
    return JSONResponse({"colab_url": colab_url})

@app.post("/api/settings")
async def save_settings(settings: SettingsModel):
    print(f"Saving AI Cloud Server URL: {settings.colab_url}")
    with open(COLAB_URL_FILE, "w", encoding="utf-8") as f:
        f.write(settings.colab_url.strip())
    return JSONResponse({"status": "success", "colab_url": settings.colab_url.strip()})

@app.post("/generate")
async def generate(photo: UploadFile = File(...), pdf: UploadFile = File(...)):
    print("\n" + "="*50)
    print("Received generation request on Local Backend!")
    os.makedirs("temp", exist_ok=True)
    
    photo_path = f"temp/{photo.filename}"
    pdf_path = f"temp/{pdf.filename}"
    out_path = f"temp/final_story.pdf"
    
    with open(photo_path, "wb") as f:
        shutil.copyfileobj(photo.file, f)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)
        
    # --- Check if Colab AI Cloud Server is Configured ---
    colab_url = ""
    if os.path.exists(COLAB_URL_FILE):
        with open(COLAB_URL_FILE, "r", encoding="utf-8") as f:
            colab_url = f.read().strip()
            
    if colab_url and colab_url.startswith("http"):
        print(f"Forwarding request to Colab AI Cloud Server: {colab_url} ...")
        try:
            with open(photo_path, "rb") as f_photo, open(pdf_path, "rb") as f_pdf:
                files = {
                    'photo': (photo.filename, f_photo, photo.content_type),
                    'pdf': (pdf.filename, f_pdf, pdf.content_type)
                }
                response = requests.post(f"{colab_url.rstrip('/')}/colab_generate", files=files, timeout=300)
                
            if response.status_code == 200:
                with open(out_path, "wb") as f_out:
                    f_out.write(response.content)
                print("Colab GPU generation successful! Returning Level 3 Masterpiece PDF.")
                return FileResponse(out_path, media_type="application/pdf", filename="Your_NexEra_Masterpiece.pdf")
            else:
                print(f"Colab server returned status {response.status_code}. Error: {response.text}")
        except Exception as e:
            print(f"Colab forwarding failed: {e}")
            print("Make sure Colab Cell 6 is running and the URL is correct!")
            
    print("Falling back to Local AI Smart Blender...")
    try:
        from smart_face_blender import process_pdf_smart
        success = process_pdf_smart(pdf_path, photo_path, out_path)
    except ImportError:
        print("smart_face_blender dependencies (insightface/onnxruntime) are not installed on this server.")
        return JSONResponse({"error": "Local generation not supported on this server. Please connect Colab GPU tunnel!"}, status_code=500)
    
    if success and os.path.exists(out_path):
        print("Local generation successful! Returning final PDF.")
        return FileResponse(out_path, media_type="application/pdf", filename="Your_NexEra_Story.pdf")
    
    return JSONResponse({"error": "Failed to process PDF"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    print("Starting NexEra AI Backend Server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
