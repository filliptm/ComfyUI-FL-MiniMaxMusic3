import asyncio

from aiohttp import web
from server import PromptServer

from ..nodes.loaders.FL_MiniMaxMusic3Loader import minimax_music3_inventory


@PromptServer.instance.routes.get("/fl/minimax-music3/status")
async def minimax_music3_status(_request):
    try:
        inventory = await asyncio.to_thread(minimax_music3_inventory)
    except (OSError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=500)
    return web.json_response(inventory)
