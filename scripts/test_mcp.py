#!/usr/bin/env python3
"""Smoke test for the FastMCP server (Task 3)."""

import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

SERVER_SCRIPT = os.path.join(_ROOT, "mcp_server.py")


async def main() -> None:
    from fastmcp import Client

    async with Client(SERVER_SCRIPT) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        assert "list_categories" in names
        assert "count_rows" in names
        print("Tools:", names)

        cats = await client.call_tool("list_categories", {})
        assert "REFUND" in str(cats)
        print("list_categories: OK")

        await client.call_tool("filter_by_intent", {"intent": "get_refund"})
        count = await client.call_tool("count_rows", {})
        assert "997" in str(count)
        print("filter_by_intent → count_rows: OK (997)")

    print("\nMCP tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
