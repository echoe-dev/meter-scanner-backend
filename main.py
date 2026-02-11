from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List
import cv2
import numpy as np
import io
import csv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    detector = cv2.barcode_BarcodeDetector()

    for file in files:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        ok, decoded_info, decoded_type, points = detector.detectAndDecode(img)

        if ok:
            for barcode in decoded_info:
                results.append({
                    "filename": file.filename,
                    "barcode": barcode
                })

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["filename", "barcode"])
    writer.writeheader()
    writer.writerows(results)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=barcodes.csv"}
    )
