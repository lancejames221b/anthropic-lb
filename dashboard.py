#!/usr/bin/env python3
"""
anthropic-lb dashboard — live monitoring UI for the load balancer.

Proxies /api/* to the LB upstream and serves a single-page dashboard
that shows routing decisions, PII redaction stats, and a test sandbox.

Author: Lance James, Unit 221B, Inc.
"""

import os
import asyncio
import aiohttp
from aiohttp import web

LB_UPSTREAM = os.environ.get("LB_UPSTREAM", "http://localhost:8891")
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))

DASHBOARD_HTML = open(
    os.path.join(os.path.dirname(__file__), "dashboard.html"), "r"
).read()


async def index_handler(request):
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def api_proxy(request):
    """Proxy /api/* requests to the LB upstream."""
    path = request.match_info.get("path", "")
    target = f"{LB_UPSTREAM}/{path}"
    method = request.method

    try:
        async with aiohttp.ClientSession() as session:
            body = await request.read() if request.can_read_body else None
            headers = {}
            if request.content_type:
                headers["Content-Type"] = request.content_type
            async with session.request(
                method, target, data=body, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.read()
                return web.Response(
                    body=data,
                    status=resp.status,
                    content_type=resp.content_type or "application/json",
                )
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)


app = web.Application()
app.router.add_get("/", index_handler)
app.router.add_route("*", "/api/{path:.*}", api_proxy)

if __name__ == "__main__":
    print(f"anthropic-lb dashboard on :{PORT} -> {LB_UPSTREAM}")
    web.run_app(app, port=PORT, print=None)
