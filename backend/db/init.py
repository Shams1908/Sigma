"""
Beanie / Motor initialisation helper.
Called from the FastAPI lifespan handler at startup.
Catches connection failures gracefully so the app still starts when the
Atlas URI is still a placeholder.
"""
from __future__ import annotations

import logging

import motor.motor_asyncio
from beanie import init_beanie

from core.config import settings
from db.models import Analysis, Signal

logger = logging.getLogger(__name__)

_DB_CONNECTED = False


async def init_db() -> bool:
    """
    Attempt to connect to MongoDB Atlas and initialise Beanie ODM.

    Returns True on success, False if the URI is a placeholder or the
    connection fails — in either case the app keeps running without
    persistence.
    """
    global _DB_CONNECTED

    placeholder = "REPLACE_ME"
    if placeholder in settings.MONGODB_URI:
        logger.warning(
            "DB not configured yet — running without persistence. "
            "Set MONGODB_URI in .env to enable MongoDB Atlas."
        )
        return False

    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
        )
        # Trigger a real network round-trip so we fail fast if the URI is bad
        await client.server_info()

        await init_beanie(
            database=client[settings.DB_NAME],
            document_models=[Signal, Analysis],
        )
        _DB_CONNECTED = True
        logger.info("MongoDB Atlas connection established (db=%s).", settings.DB_NAME)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "DB not configured yet — running without persistence. "
            "MongoDB connection failed: %s",
            exc,
        )
        return False


def is_db_connected() -> bool:
    return _DB_CONNECTED
