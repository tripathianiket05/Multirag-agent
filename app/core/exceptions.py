import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class MultiRAGException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

async def multirag_exception_handler(request: Request, exc: MultiRAGException):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("%s | request_id=%s", exc.message, request_id)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "request_id": request_id})