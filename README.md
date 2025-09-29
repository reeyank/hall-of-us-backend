# 🏛️ Hall of Us Backend — Modular, Agentic, and LangChain-Inspired

This backend powers **Hall of Us** — a museum-inspired social memories app built at HackGT! It is not just a simple API: it is a modular, agentic, and extensible backend inspired by LangChain, with custom FastAPI types, robust Pydantic v2 models, and a framework for building AI-driven, stateful, and composable API operations.

**Key highlights:**

- Modular, agentic architecture for rapid extension and experimentation
- Custom FastAPI types and Pydantic v2 models for every endpoint
- LangChain-inspired API wrapper for unified, composable AI calls
- Designed for hackathon speed _and_ production extensibility

## 🚀 Features

- **Modular, Agentic Framework**: Each API operation is a modular agent, making it easy to add, compose, or swap out logic. The backend is designed for agentic workflows and composable chains of logic.
- **Custom FastAPI Types**: Every endpoint uses custom, strongly-typed Pydantic v2 models, ensuring robust request/response validation and seamless frontend/backend integration.
- **LangChain-Inspired API Wrapper**: Unified interface for all AI calls (OpenAI, Google Vision, etc.), with built-in error handling, retries, logging, and batch processing.
- **OpenAI Vision & Chat**: AI-powered tagging, captioning, and natural language filtering, with vision endpoints that can process both URLs and base64 images.
- **Cedar OS Integration**: Handles complex Cedar OS state, including `allMemories`, and only reads/sends state to OpenAI when required by the user’s request.
- **S3/R2 Uploads**: Secure image upload, EXIF extraction, and cloud storage support.
- **Dynamic Filtering**: Natural language to backend filter generation, with LLM-driven logic (no rule-based fallback).
- **Streaming Support**: Real-time streaming completions for chat and vision endpoints.
- **Modern Python**: Async, type-annotated, modular, and hackathon-friendly.

## 🛠️ Endpoints (All with Custom Types & Modular Handlers)

- `POST /langchain/chat/completions` — Chat/vision completions (streaming or regular)
- `POST /langchain/chat/generate-tags` — Generate tags for an image (URL, base64, or file)
- `POST /langchain/chat/fill-tags` — Suggest additional tags
- `POST /langchain/chat/generate-caption` — Generate a caption for an image
- `POST /langchain/chat/fill-caption` — Enhance a caption
- `POST /langchain/chat/filter_images` — AI-powered image filtering (tags, users, natural language)
- `POST /photos/upload` — Upload a photo (with frame, EXIF, and S3/R2 support)
- `GET /photos` — List all photos
- `GET /photos/{image_id}` — Get a photo by ID
- `POST /photos/{photo_id}/like` — Like a photo
- `POST /photos/{photo_id}/unlike` — Unlike a photo
- `GET /gps` — Get GPS data for all photos
- `GET /plaque_text` — Generate a plaque image with text
- `POST /signup` — User signup
- `POST /signin` — User signin

## 🧠 LangChain, Agentic, and AI State Handling

- **Agentic Modular Design**: Each API operation is an agent — easy to extend, compose, and orchestrate. Add new endpoints or swap logic with minimal friction.
- **Custom FastAPI/Pydantic Types**: All requests and responses use custom, strongly-typed models for robust validation and clear API contracts.
- **LangChain-Inspired API Wrapper**: All AI calls (OpenAI, Google Vision, etc.) go through a unified, composable wrapper with error handling, retries, and logging.
- **OpenAI Vision**: When a user asks about images, the backend extracts image URLs from Cedar state (e.g., `allMemories`) and sends them to OpenAI for analysis.
- **Smart State Reading**: The backend only reads Cedar state if the user's request requires it (e.g., "show me my images"). By default, only the first 5 memories are included; if the user explicitly asks for "all" images, the entire state is used.
- **Natural Language Filtering**: The `/filter_images` endpoint uses OpenAI to generate backend filters from tags, users, and freeform text, with no rule-based fallback.

## 🏗️ Project Structure (Modular & Extensible)

```
main.py                 # FastAPI app entrypoint
langchain_router.py     # FastAPI router with all endpoints (modular wiring)
langchain/              # Core backend logic (agentic modules)
    chat_handlers.py      # Modular AI chat, tagging, captioning, filtering logic (agentic)
    response_utils.py     # Image extraction, response formatting
    models.py             # Custom Pydantic v2 models for all endpoints
    shared_context.py     # In-memory chat/session state
    wrapper.py            # LangChain-inspired API wrapper (unified AI calls)
    ...
s3_upload.py            # S3/R2 upload helpers
requirements.txt        # Python dependencies
.env                    # Environment variables (OpenAI, S3, DB, etc)
```

## ⚡ Quickstart

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up your `.env` file:**

```
OPENAI_API_KEY=sk-...
S3_BUCKET_NAME=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=...
```

3. **Run the backend:**

```bash
uvicorn main:app --reload
```

4. **Connect the frontend:**

- The frontend expects the backend at `http://localhost:8000` by default.
- See the frontend README for more details on API usage and integration.

## 🤝 Integration with Frontend

- Designed to work seamlessly with the Hall of Us Next.js/Cedar OS frontend.
- Handles Cedar state, AI chat, and image analysis requests.
- All API types and contracts are custom and robust, making frontend/backend integration smooth.
- See the frontend repo for API call examples and state structure.

### 🪢 Sister Repositories
- [hall-of-us-frontend (Frontend API)](https://github.com/reeyank/hall-of-us-backend)
- [Hall-Of-Us-XR (XR Experience)](https://github.com/zaid-ahmed1/Hall-Of-Us-XR/)


## � Notes

- Modular, agentic, and extensible: add new endpoints or swap logic with minimal code changes.
- Only reads and sends image state to OpenAI if the user request requires it ("show me my images").
- For vision endpoints, only the first 5 images are sent unless the user explicitly asks for all.
- All AI filtering is LLM-driven (no rule-based fallback).
- For hackathon/demo use: not production-hardened, but easy to extend!

---

Made with ❤️ at HackGT
