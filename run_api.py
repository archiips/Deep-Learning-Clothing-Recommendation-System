"""
Simple script to run the FastAPI server locally.
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    print("=" * 60)
    print("Starting Clothing Recommendation API")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://localhost:{port}/docs")
    print(f"Health: http://localhost:{port}/health")
    print("=" * 60)

    # Run server with auto-reload for development
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
