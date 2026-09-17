import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting FraudGraph server on http://127.0.0.1:{port}")
    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=port, reload=False)
