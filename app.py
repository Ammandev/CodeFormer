from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import cv2
from enhancer import inference
from config import configure_app

app = FastAPI(
    title="CodeFormer API",
    description="API for CodeFormer: Robust Face Restoration and Enhancement Network"
)

# Configure the app with our custom settings
configure_app(app)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>CodeFormer API</title>
        </head>
        <body>
            <h1>CodeFormer API</h1>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><a href="/redoc">/redoc</a> - Alternative API documentation</li>
                <li><code>POST /restore/</code> - Endpoint to restore images</li>
            </ul>
        </body>
    </html>
    """

class RestoreParams(BaseModel):
    face_align: bool = True
    background_enhance: bool = True
    face_upsample: bool = True
    upscale: float = 2.0
    codeformer_fidelity: float = 0.5

@app.post("/restore/")
async def restore_image(
    file: UploadFile = File(...),
    face_align: bool = True,
    background_enhance: bool = True,
    face_upsample: bool = True,
    upscale: float = 2.0,
    codeformer_fidelity: float = 0.5
):
    """
    Restore a face image using CodeFormer
    """
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process image
        restored_img = inference(
            temp_file,
            face_align,
            background_enhance,
            face_upsample,
            upscale,
            codeformer_fidelity
        )
        
        # Save result
        output_path = f'output/restored_{file.filename}'
        os.makedirs('output', exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))
        
        # Clean up temp file
        os.remove(temp_file)
        
        # Return the processed image
        return FileResponse(output_path)
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=10,
        timeout_keep_alive=120
    )