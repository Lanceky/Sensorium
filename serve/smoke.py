"""End-to-end check of the reasoning service against a phone-shaped payload.

Run from the repo root with the service already listening::

    python -m serve.api &
    python -m serve.smoke

This is a smoke test, not a measurement. The numbers it reports come from one live run,
which is exactly why it is kept out of the results table: the harness in ``eval/`` is what
produces claims, and this only answers "does the app's payload shape survive the whole
workflow". It builds the same JSON the Android client sends, so a mismatch between the
phone's ``device_slice`` and the engine's contract fails here rather than on stage.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import sys
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8765"


def device_slice(weeks: int = 4, rising: bool = True) -> dict:
    """Build the payload SignalStore.seedHistory produces on the phone.

    The drift runs backwards from today's readings for the same reason the Kotlin does it:
    the series has to end where the device actually is, or the last point contradicts the
    panel the person just looked at. Noise is added deliberately - a perfectly straight line
    makes the significance test meaningless, and a demo whose p-value is an artefact of
    noiseless data is demonstrating the generator, not the engine.
    """
    random.seed(11)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    events = []

    for week in range(weeks):
        days_back = (weeks - 1 - week) * 7
        stamp = (now - dt.timedelta(days=days_back)).isoformat()
        step = week if rising else 0
        events.append({
            "signal": "volume",
            "ts": stamp,
            "value": round(7 + step * 1.5 + random.uniform(-0.4, 0.4), 2),
        })
        events.append({
            "signal": "brightness",
            "ts": stamp,
            "value": round(120 + step * 14 + random.uniform(-6, 6), 2),
        })
        events.append({
            "signal": "font_scale",
            "ts": stamp,
            "value": round(1.0 + step * 0.05 + random.uniform(-0.01, 0.01), 3),
        })
        events.append({"signal": "caption", "ts": stamp, "value": 1.0 if step >= 2 else 0.0})

    return {
        "events": events,
        "profile_id": "device-local",
        "window": {
            "start": (now - dt.timedelta(days=(weeks - 1) * 7)).date().isoformat(),
            "end": now.date().isoformat(),
        },
    }


CONVERSATION = [
    {"role": "agent", "text": "Anything felt off with your eyes or ears lately?"},
    {"role": "user", "text": "I keep asking people to repeat themselves in the cafeteria."},
    {"role": "agent", "text": "Was that once, or has it come up a few times?"},
    {"role": "user", "text": "Maybe four or five times in the last couple of weeks."},
]


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> int:
    with urllib.request.urlopen(BASE_URL + "/health", timeout=10) as response:
        print("health:", json.loads(response.read()))

    try:
        result = post("/analyse", {
            "device_slice": device_slice(),
            "conversation": CONVERSATION,
        })
    except urllib.error.HTTPError as error:
        print("REFUSED:", error.read().decode()[:2000])
        return 1

    print("\nrun:", result["run_id"])
    print("headline:", result["headline"])
    print("insufficient_data:", result["insufficient_data"])
    print("disagreement:", result["disagreement"])
    print("\nfigures:")
    for figure in result["figures"]:
        print(f"  {figure['name']:<34} {figure['value']:<18} significant={figure['significant']}")
    print("\nsuggestions:")
    for suggestion in result["suggestions"]:
        print(f"  - {suggestion['text']}\n    source={suggestion['source_url']}")
    print("\nobservations:", len(result["observations"]),
          "| no-symptom:", len(result["no_symptom_statements"]))
    print("\nreport:\n", result["report_markdown"][:1400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
