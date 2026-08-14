import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_tool(server_path, tool_name, arguments, session_id):
    environment = os.environ.copy()
    environment.update({
        "FL_MCP_MODE": "subprocess",
        "FL_MCP_SESSION_ID": session_id,
        "FL_MCP_WS_URL": "ws://127.0.0.1:8000/ws",
    })
    parameters = StdioServerParameters(
        command=str(Path(__file__).parents[3] / "venv" / "Scripts" / "python.exe"),
        args=[str(server_path)],
        env=environment,
    )
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as client:
            await client.initialize()
            if tool_name == "list_tools":
                result = await client.list_tools()
                return [
                    {"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema}
                    for tool in result.tools
                ]
            result = await client.call_tool(tool_name, arguments)
            return result.model_dump(mode="json", exclude_none=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("tool")
    parser.add_argument("arguments", nargs="?", default="{}")
    parser.add_argument("--arguments-base64")
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--server",
        default=str(Path(__file__).parents[2] / "ComfyUI_FL-MCP" / "backend" / "mcp_server.py"),
    )
    args = parser.parse_args()
    arguments = args.arguments
    if args.arguments_base64:
        arguments = base64.b64decode(args.arguments_base64).decode("utf-8")
    payload = asyncio.run(call_tool(args.server, args.tool, json.loads(arguments), args.session))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
