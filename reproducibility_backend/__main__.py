import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "reproducibility_backend.app:app",
        host="127.0.0.1",
        port=8768,
        reload=False,
    )
