"""FastAPI app assembly.

Route order matters: the catch-all fallback below must be registered last so
it only catches requests that didn't match a real route.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import api_inference, api_meta

app = FastAPI(title="ov-ollama")

app.include_router(api_meta.router)
app.include_router(api_inference.router)


# --------------------------------------------------------------------------- #
# Fallback: anything else Ollama's real API supports but this server doesn't
# (e.g. /api/pull, /api/embed, /api/copy, /api/delete, /api/create).
# Registered last, so it only catches requests that didn't match a route
# above - logs to stdout/server.log instead of failing silently.
# --------------------------------------------------------------------------- #


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def not_implemented(full_path: str, request: Request):
    path = "/" + full_path
    print(f"[UNIMPLEMENTED] {request.method} {path} was requested but is not supported by this server", flush=True)
    return JSONResponse(
        {"error": f"{path} is not implemented by this minimal OpenVINO-backed Ollama server"},
        status_code=404,
    )
