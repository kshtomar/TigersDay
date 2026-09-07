"""
Tiger's Day – Local Development & AI Server
Serves both the FastAPI REST backend and static browser client.
"""

import os
import sys
import argparse
import uvicorn

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.app import create_app

app = create_app(mount_static=True)

def main():
    parser = argparse.ArgumentParser(
        description="Serve the Tiger's Day game UI and AI backend."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    print(f"🐅 Tiger's Day Server starting on http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()