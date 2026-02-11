from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import cv2
import numpy as np
import io

app = FastAPI()

# ✅ CORS setup
origins = [
    "http://localhost:5500",  # your local frontend
    "http://127.0.0.1:5500",
    "https://your-frontend-domain.com"  # add your deployed frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Server running"}

@app.post("/scan")
async def scan(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        try:
            # Read the file content into numpy array
            contents = await file.read()
            np_array = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if img is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to decode image {file.filename}"}
                )

            # Example: get image dimensions
            height, width, channels = img.shape
            results.append({
                "filename": file.filename,
                "height": height,
                "width": width,
                "channels": channels
            })

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed processing {file.filename}: {str(e)}"}
            )

    return {"status": "success", "files": results}
