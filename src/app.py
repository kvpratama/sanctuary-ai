from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.db.client import close_supabase_client
from src.routers import chat, ingest

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown resources."""
    yield
    await close_supabase_client()


app = FastAPI(lifespan=lifespan)


def _add_cors_middleware(app: FastAPI) -> None:
    """Configure CORS middleware using settings."""
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,  # type: ignore
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_add_cors_middleware(app)


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
