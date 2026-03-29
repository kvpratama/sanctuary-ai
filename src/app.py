from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import chat, ingest

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root() -> dict[str, str]:
    """Return a welcome message for the API.

    Returns:
        A dictionary containing the API message.
    """
    return {"message": "Bookified API"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Check the health status of the API.

    Returns:
        A dictionary containing the health status.
    """
    return {"status": "ok"}


app.include_router(chat.router)
app.include_router(ingest.router)
