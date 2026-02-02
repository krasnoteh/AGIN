from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from agin.servers.websocket_server.router import router, get_stream_processor_service
from agin.servers.websocket_server.stream_processor_service import StreamProcessorService

def main():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    stream_processor_service = StreamProcessorService(config_path="configs/stream_processor_config.json")
    app.dependency_overrides[get_stream_processor_service] = lambda: stream_processor_service
    app.include_router(router, prefix="/services/agin")

if __name__ == "__main__":
    main()