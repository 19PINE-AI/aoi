#!/usr/bin/env python3
"""Protocol smoke test for the OpenAI Realtime websocket API (gpt-realtime-2).

Goal: confirm (a) we can open a session, (b) the model accepts an image +
text turn, and (c) it emits a real function/tool call. Prints every server
event so we can lock down the exact GA session schema before wiring the
browser adapter. No browser, no audio yet.
"""
import base64
import io
import json
import os
import sys
from pathlib import Path

import websocket  # websocket-client (sync)
from PIL import Image

MODEL = os.environ.get("RT_MODEL", "gpt-realtime-2")
KEY = os.environ["OPENAI_RT_KEY"]
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

TOOLS = [
    {
        "type": "function",
        "name": "fill",
        "description": "Fill a form input identified by HTML id with the given text.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
            "required": ["id", "text"],
        },
    },
    {
        "type": "function",
        "name": "wait",
        "description": "Do nothing this step; observe more.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def make_test_image() -> str:
    """A trivial image with a visible form id so a vision model has something to act on."""
    img = Image.new("RGB", (640, 200), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ws = websocket.create_connection(
        URL,
        header=[f"Authorization: Bearer {KEY}"],  # GA API: no beta header
        timeout=30,
    )
    print("CONNECTED")

    def send(ev):
        ws.send(json.dumps(ev))

    # Try GA session schema first; fall back is driven by inspecting errors.
    send({
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": "You are a computer-use agent. Call exactly one tool.",
            "tools": TOOLS,
            "tool_choice": "required",
            "output_modalities": ["text"],
        },
    })

    # One user turn: image + instruction.
    send({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": "Answer the form field #answer with the text 'hello'. Pick one tool."},
                {"type": "input_image",
                 "image_url": f"data:image/jpeg;base64,{make_test_image()}"},
            ],
        },
    })
    send({"type": "response.create"})

    got_tool = None
    n = 0
    while n < 60:
        n += 1
        try:
            raw = ws.recv()
        except Exception as e:
            print("RECV ERROR:", e)
            break
        if not raw:
            continue
        ev = json.loads(raw)
        t = ev.get("type", "?")
        # Compact print of each event type + salient fields
        if t == "error":
            print("EVENT error:", json.dumps(ev.get("error", ev))[:400])
        elif t in ("response.function_call_arguments.done",):
            print(f"EVENT {t}: name={ev.get('name')} args={ev.get('arguments')}")
            got_tool = (ev.get("name"), ev.get("arguments"))
        elif t == "response.output_item.done":
            item = ev.get("item", {})
            print(f"EVENT {t}: item.type={item.get('type')} name={item.get('name')} args={str(item.get('arguments'))[:120]}")
            if item.get("type") == "function_call":
                got_tool = (item.get("name"), item.get("arguments"))
        elif t == "response.done":
            resp = ev.get("response", {})
            print(f"EVENT response.done: status={resp.get('status')} "
                  f"status_detail={json.dumps(resp.get('status_details'))[:200]}")
            break
        elif t in ("session.created", "session.updated"):
            print(f"EVENT {t}")
        else:
            print(f"EVENT {t}")

    ws.close()
    print("\nRESULT got_tool =", got_tool)
    sys.exit(0 if got_tool else 2)


if __name__ == "__main__":
    main()
