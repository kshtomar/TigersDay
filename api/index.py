"""
Tiger's Day – Serverless API Entrypoint for Vercel
Imports the unified FastAPI app factory from api.app
"""

import os
import sys

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.app import create_app

app = create_app(mount_static=False)