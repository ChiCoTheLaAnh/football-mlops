# Cloud deployment guide

This repository is ready for free containerized deployment on Render (recommended) and Railway/Fly.io. The API is a FastAPI app served by Uvicorn from `src/api/app.py`.

## Prerequisites
- A GitHub repository containing this code (Render pulls directly from a repo)
- Render account (free tier works) or similar Docker-based hosting provider
- The bundled `models/` directory is small enough for free tiers, so no extra storage is required

## Deploy to Render (Docker)
1. Push your fork of this repo to GitHub.
2. In Render, choose **New > Web Service** and connect the repo.
3. When prompted for the Blueprint, select **Use Render.yaml**; Render will read `render.yaml` at the repo root.
4. Confirm the following settings (already defined in `render.yaml`):
   - **Environment:** Docker
   - **Health Check Path:** `/health`
   - **Branch:** `main` (update `render.yaml` if you use a different branch)
5. Deploy. Render injects the `PORT` environment variable automatically; the Dockerfile now uses it when starting Uvicorn.
6. After the service is live, grab the public URL from the Render dashboard.

### Smoke test the deployed API
Replace `<render-url>` with your service URL.

```bash
curl -s https://<render-url>/health
curl -s -X POST https://<render-url>/predict/xg \
  -H 'Content-Type: application/json' \
  -d '{"shots": [{"distance_to_goal": 12.3, "shot_angle": 0.4, "body_part": "left_foot", "shot_type": "open_play"}]}'
```

## Railway/Fly.io
Both platforms also pass a `PORT` variable to containers. The updated Dockerfile respects `PORT`, so you can deploy with their default Docker workflows. Point any health checks at `/health`.

## Public demo URL
This environment cannot create external cloud resources, so no live URL is attached. Follow the steps above on your cloud account to publish the API URL.
