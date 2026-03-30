import os

# Set dummy environment variables for tests to avoid pydantic-settings ValidationError
# when testing in an environment without a .env file (e.g., GitHub Actions CI).
os.environ.setdefault("SUPABASE_URL", "http://localhost:8000")
os.environ.setdefault("SUPABASE_KEY", "dummy-key")
os.environ.setdefault("SUPABASE_ANON_KEY", "dummy-anon-key")
os.environ.setdefault("SUPABASE_JWT_SECRET", "dummy-jwt")
os.environ.setdefault("OPENAI_API_KEY", "dummy-openai-key")
os.environ.setdefault("GEMINI_API_KEY", "dummy-gemini-key")
os.environ.setdefault("BOOKIFIED_BLOB_READ_WRITE_TOKEN", "dummy-blob-token")
