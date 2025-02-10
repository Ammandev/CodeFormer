import sys
sys.path.append('CodeFormer')
import os
import cv2
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import numpy as np
from pydantic import BaseModel
from typing import Optional
import uvicorn

from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY


os.system("pip freeze")

pretrain_model_url = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
}
# download weights
for key, url in pretrain_model_url.items():
    if key == 'codeformer' and not os.path.exists('CodeFormer/weights/CodeFormer/codeformer.pth'):
        load_file_from_url(url=url, model_dir='CodeFormer/weights/CodeFormer', progress=True, file_name=None)
    elif key == 'detection' and not os.path.exists('CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
        load_file_from_url(url=url, model_dir='CodeFormer/weights/facelib', progress=True, file_name=None)
    elif key == 'parsing' and not os.path.exists('CodeFormer/weights/facelib/parsing_parsenet.pth'):
        load_file_from_url(url=url, model_dir='CodeFormer/weights/facelib', progress=True, file_name=None)
    elif key == 'realesrgan' and not os.path.exists('CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth'):
        load_file_from_url(url=url, model_dir='CodeFormer/weights/realesrgan', progress=True, file_name=None)
def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# set enhancer with RealESRGAN
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler

upsampler = set_realesrgan()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to(device)
ckpt_path = "CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()

os.makedirs('output', exist_ok=True)

def inference(image, face_align, background_enhance, face_upsample, upscale, codeformer_fidelity):
    """Run a single prediction on the model"""
    try: # global try
        # take the default setting for the demo
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        print('Inp:', image, background_enhance, face_upsample, upscale, codeformer_fidelity)
        face_align = face_align if face_align is not None else True
        background_enhance = background_enhance if background_enhance is not None else True
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        has_aligned = not face_align
        upscale = 1 if has_aligned else upscale

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        print('\timage size:', img.shape)

        upscale = int(upscale) # convert type to int
        if upscale > 4: # avoid memory exceeded due to too large upscale
            upscale = 4 
        if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
            upscale = 2 
        if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            if face_helper.is_gray:
                print('\tgrayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )
        else:
            restored_img = restored_face

        # save restored img
        save_path = f'output/out.png'
        imwrite(restored_img, str(save_path))

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print('Global exception', error)
        return None, None

app = FastAPI(
    title="CodeFormer API",
    description="API for CodeFormer: Robust Face Restoration and Enhancement Network"
)
# Add a root route
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
        img = cv2.imread(temp_file)
        
        # Run inference using the existing inference function
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
    uvicorn.run(app, host="0.0.0.0", port=8000)