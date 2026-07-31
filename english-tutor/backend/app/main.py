from fastapi import FastAPI

app = FastAPI(title="English Tutor API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
