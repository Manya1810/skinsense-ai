"""SkinSense AI - Application Entry Point
Hosts the modern clinical web application and multi-agent AI backend.
"""

import os
from pathlib import Path
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print("\n" + "=" * 60)
    print("SkinSense AI 2.0 - Multimodal Clinician Decision Support")
    print(f"Server active at: http://{host}:{port}")
    print("=" * 60 + "\n")
    uvicorn.run("web_app:app", host=host, port=port, reload=False)
