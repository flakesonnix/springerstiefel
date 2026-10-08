#!/usr/bin/env python3
"""Headless-manual capture of the real Hey_ requests into data/traffic.jsonl."""

import asyncio
import json
from pathlib import Path
from typing import Any

from playwright.async_api import Request, Response, async_playwright

DATA_DIR = Path("data")
PROFILE_DIR = DATA_DIR / "browser-firefox"
TRAFFIC_FILE = DATA_DIR / "traffic.jsonl"

INTERESTING_KEYWORDS: tuple[str, ...] = (
    "api",
    "chat",
    "conversation",
    "message",
    "completion",
    "graphql",
    "experience",
)


def is_interesting_url(url: str) -> bool:
    return any(x in url.lower() for x in INTERESTING_KEYWORDS)


async def main_async() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        context = await p.firefox.launch_persistent_context(
            str(PROFILE_DIR),
            headless=False,
        )

        page = context.pages[0] if context.pages else await context.new_page()

        print("[*] Opening Hey_ …")
        await page.goto("https://hey.bild.de/", wait_until="domcontentloaded")

        print()
        print("==============================================")
        print(" Browser is open.")
        print(" Log in if needed.")
        print(" Then send a test message in Hey_.")
        print(" Example: 'Say hello'")
        print("==============================================")
        print()

        async def request_handler(request: Request) -> None:
            resource: str = request.resource_type

            if resource not in {"fetch", "xhr"}:
                return

            url: str = request.url

            if not is_interesting_url(url):
                return

            post_data: str | None
            try:
                post_data = request.post_data
            except Exception:
                post_data = None

            entry: dict[str, Any] = {
                "kind": "request",
                "method": request.method,
                "url": url,
                "resource_type": resource,
                "headers": dict(request.headers),
                "post_data": post_data,
            }

            with TRAFFIC_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print()
            print(">>> REQUEST")
            print(request.method, url)

            if post_data:
                print(post_data[:4000])

        async def response_handler(response: Response) -> None:
            request = response.request

            if request.resource_type not in {"fetch", "xhr"}:
                return

            url: str = response.url

            if not is_interesting_url(url):
                return

            body: str
            try:
                body = await response.text()
            except Exception:
                body = "<unable to read response>"

            entry: dict[str, Any] = {
                "kind": "response",
                "status": response.status,
                "url": url,
                "headers": dict(response.headers),
                "body": body[:100000],
            }

            with TRAFFIC_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print()
            print("<<< RESPONSE", response.status, url)
            print(body[:3000])

        page.on("request", request_handler)
        page.on("response", response_handler)

        print("[*] Capture running.")
        print("[*] Press Enter once you sent the test message.")
        await asyncio.to_thread(input)

        print()
        print(f"[*] Traffic saved to: {TRAFFIC_FILE}")

        await context.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
