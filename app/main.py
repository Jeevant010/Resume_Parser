from fastapi import FastAPI
from app.api.endpoints import router as api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Modern Resume Parser API",
    description="A Zero-Shot ML-powered API for extracting Named Entities from Resumes.",
    version="1.0.0"
)

# Allow cross-origin requests from frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "message": "Welcome to the Modern Resume Parser API.",
        "docs": "Visit /docs for the interactive Swagger UI.",
        "status": "Online"
    }

# To run: uvicorn app.main:app --reload
