# SkinSense AI — Complete Cloud & Local Deployment Guide

This document outlines the step-by-step procedures for deploying **SkinSense AI** across popular cloud hosting providers, containerized setups (Docker), and free public demo platforms.

---

## 🚀 Option 1: Hugging Face Spaces (Recommended for Free Public Medical Demos)

Hugging Face Spaces provides free hosting with CPU / GPU instances, perfect for machine learning portfolio demonstrations.

### Steps:
1. Log in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Space Name: `skinsense-ai`
3. Space SDK: Select **Docker** (Blank template).
4. Space Hardware: Free (2 vCPU, 16GB RAM) or CPU Upgrade.
5. In your local terminal, initialize a git repository and push:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<your-username>/skinsense-ai
   git add .
   git commit -m "Deploy SkinSense AI"
   git push -u origin main
   ```
6. The `Dockerfile` in the root directory will automatically build the environment, install FFmpeg, PyTorch, CrewAI, and Google ADK, and launch the web interface on port `7860`.

---

## 🌐 Option 2: Render.com (Free Web Service)

Render allows easy deployment directly from a GitHub repository.

### Steps:
1. Push your repository to **GitHub**.
2. Go to [Render.com](https://render.com/) and create a **New Web Service**.
3. Connect your GitHub repository.
4. Settings:
   - **Environment:** `Docker` (Render will automatically detect `Dockerfile`)
   - **Plan:** Free
   - **Environment Variables (Optional):**
     - `PORT`: `7860`
     - `HOST`: `0.0.0.0`
     - `GROQ_API_KEY`: `<your_key>` (Optional)
5. Click **Deploy Web Service**.
6. Once deployed, Render will provide a public URL: `https://skinsense-ai.onrender.com`.

---

## ⚡ Option 3: Railway.app

Railway provides ultra-fast Docker deployments with built-in HTTPS domains.

### Steps:
1. Push code to GitHub.
2. Log into [Railway.app](https://railway.app/).
3. Click **New Project** -> **Deploy from GitHub repo**.
4. Select your `ai-skin-specialist` repository.
5. In **Settings** -> **Networking**, click **Generate Domain**.
6. Railway automatically detects `Dockerfile` and maps port `7860`.

---

## 🐳 Option 4: Local or VPS Docker Deployment

To run containerized locally or on any Linux VPS (Ubuntu / AWS EC2 / DigitalOcean Droplet / Google Cloud Compute Engine):

### Build and Run using Docker Compose:
```bash
docker compose up -d --build
```
The app will be accessible at `http://localhost:7860`.

### Or using standard Docker CLI:
```bash
# Build the container
docker build -t skinsense-ai .

# Run the container
docker run -d -p 7860:7860 --name skinsense-container skinsense-ai
```

---

## 💻 Option 5: Local Production Execution (Windows / Mac / Linux)

To run directly on a local server or presentation laptop:
```bash
# Install dependencies
pip install -r requirements.txt

# Run main server
python main.py
```
Access at: `http://localhost:7860`.
