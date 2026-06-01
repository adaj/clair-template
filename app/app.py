from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router


with open('app/README.md', 'r') as f:
    description = f.read()

app = FastAPI(
    title="Clair API",
    description=description,
    version="0.4"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=[
    #     "https://apps.dev.graasp.eu",
    #     "http://apps.dev.graasp.eu",
    #     "https://utwente.graasp.eu",
    #     "http://go-lab.bms.utwente.nl",
    #     "https://go-lab.bms.utwente.nl",
    #     "http://go-lab-develop.bms.utwente.nl",
    #     "https://go-lab-develop.bms.utwente.nl",
    # ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Clair API. Check /redoc for more info."
    }
