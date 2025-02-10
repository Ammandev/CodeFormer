from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import cv2
from enhancer import inference
import logging
import time
from datetime import datetime
from config import configure_app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        start_time = time.time()
        logger.info(f"Starting image restoration process for file: {file.filename}")
        logger.info(f"Parameters: align={face_align}, bg_enhance={background_enhance}, "
                   f"face_upsample={face_upsample}, upscale={upscale}, fidelity={codeformer_fidelity}")

        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        logger.debug(f"Saving temporary file: {temp_file}")
        file_save_start = time.time()
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.debug(f"File save completed in {time.time() - file_save_start:.2f} seconds")
        
        # Process image
        logger.info("Starting inference process")
        inference_start = time.time()
        restored_img = inference(
            temp_file,
            face_align,
            background_enhance,
            face_upsample,
            upscale,
            codeformer_fidelity
        )
        inference_time = time.time() - inference_start
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        if restored_img is None:
            logger.error("Inference failed - restored_img is None")
            raise Exception("Image restoration failed")

        # Save result
        save_start = time.time()
        output_path = f'output/restored_{file.filename}'
        os.makedirs('output', exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))
        logger.debug(f"Result saved to {output_path} in {time.time() - save_start:.2f} seconds")
        
        # Clean up temp file
        logger.debug(f"Removing temporary file: {temp_file}")
        os.remove(temp_file)
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Return the processed image
        return FileResponse(output_path)
    
    except Exception as e:
        logger.error(f"Error during image restoration: {str(e)}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting CodeFormer API server")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=10,
        timeout_keep_alive=120
    )