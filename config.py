from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import json

class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int = 100 * 1024 * 1024):  # 100MB default
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.method == 'POST':
            if 'content-length' in request.headers:
                content_length = int(request.headers['content-length'])
                if content_length > self.max_upload_size:
                    return Response(
                        content=json.dumps({
                            "error": f"File too large. Maximum size allowed is {self.max_upload_size/1024/1024}MB"
                        }),
                        status_code=413
                    )
        return await call_next(request)

def configure_app(app: FastAPI):
    # Add middleware to limit upload size (100MB)
    app.add_middleware(LimitUploadSize, max_upload_size=100 * 1024 * 1024)
    
    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=['*']  # In production, replace with your actual domains
    ) 