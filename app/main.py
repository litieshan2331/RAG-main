import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.mineru_process import MinerUProcessManager
from app.services.ingestion_tasks import IngestionTaskManager


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    app.include_router(router)
    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    mineru_manager = MinerUProcessManager()
    ingestion_manager = IngestionTaskManager()
    app.state.mineru_manager = mineru_manager
    app.state.ingestion_manager = ingestion_manager
    await asyncio.to_thread(ingestion_manager.start)
    await mineru_manager.start()
    try:
        yield
    finally:
        ingestion_manager.shutdown()
        await mineru_manager.stop()


app = create_app()
