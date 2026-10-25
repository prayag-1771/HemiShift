"""HemiShift live demo: stream a never-seen stroke patient over LSL and decode them live.

Three pieces run together:
  replay    publishes the patient's raw 64-channel EEG and cue markers on LSL, like an amplifier
  receiver  decodes that stream: calibrates on the first k (rest, MI) pairs, then decodes every
            trial with the patient's own decoder and with the pooled HemiShift decoder
  server    a dashboard at http://127.0.0.1:<port> showing the stream and both decoders live
The default patient is from the S3 locked test, never used while developing the method.

Usage: .venv/Scripts/python -I demo/server.py [--patient 22] [--speed 2] [--k 5] [--port 8000]
"""

import argparse
import asyncio
import json
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).parent))
from donors import load_donors  # noqa: E402

from hemishift.data.loaders import wang2026sf_affected_side  # noqa: E402
from hemishift.online.lsl import Receiver, Replay, wang2026sf_recording  # noqa: E402
from hemishift.online.session import Session  # noqa: E402
from hemishift.paths import RESULTS_DIR  # noqa: E402

STATIC = Path(__file__).parent / "static"


class Hub:
    """Fans events out from the decoder thread to every connected browser. Everything but raw
    EEG is kept, so a browser that connects late (or reloads) catches up."""

    def __init__(self):
        self.loop, self.clients, self.history = None, set(), []

    def publish(self, event):  # safe from any thread
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event):
        if event["type"] != "eeg":
            self.history.append(event)
        for queue in list(self.clients):
            queue.put_nowait(event)


def offline_reference(patient, k):
    """This patient's offline locked-test accuracies, for the dashboard's reference line."""
    path = RESULTS_DIR / "summary" / "s3_locked_test.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    row = table[(table["subject"] == patient) & (table["k"] == k)]
    return None if row.empty else {"own": float(row["own"].iloc[0]), "pooled": float(row["pooled"].iloc[0])}


def create_app(args):
    print(f"loading patient {args.patient} and the donor pool ...")
    data, names, sfreq, events = wang2026sf_recording(args.patient)
    donor_C, donor_y = load_donors()
    side = wang2026sf_affected_side()[args.patient]
    meta = {"type": "meta", "patient": args.patient, "affected_side": side, "k": args.k,
            "speed": args.speed, "sfreq": sfreq, "channels": len(names),
            "duration_s": data.shape[1] / sfreq, "n_cues": len(events),
            "offline": offline_reference(args.patient, args.k), "donor_patients": 16,
            "donor_epochs": int(len(donor_y))}
    hub = Hub()
    state = {"started": False, "threads": []}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        hub.loop = asyncio.get_running_loop()
        yield
        for thread in state["threads"]:
            thread.stop_flag.set()

    app = FastAPI(title="HemiShift live demo", lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/start")
    async def start():
        with lock:
            if not state["started"]:
                state["started"] = True
                hub.publish({"type": "started"})
                receiver = Receiver(Session(donor_C, donor_y, k=args.k), side, hub.publish)
                replay = Replay(data, names, sfreq, events, speed=args.speed)
                state["threads"] = [receiver, replay]
                receiver.start()
                replay.start()
        return {"started": True}

    @app.get("/events")
    async def stream_events():
        queue = asyncio.Queue()
        hub.clients.add(queue)

        async def stream():
            try:
                yield f"data: {json.dumps(meta)}\n\n"
                for event in list(hub.history):
                    yield f"data: {json.dumps(event)}\n\n"
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                hub.clients.discard(queue)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main():
    parser = argparse.ArgumentParser(description="HemiShift live demo")
    parser.add_argument("--patient", type=int, default=22, help="Tianjin subject number (locked test: 1 4 5 6 8 15 17 22)")
    parser.add_argument("--speed", type=float, default=2.0, help="replay speed (1 = real time)")
    parser.add_argument("--k", type=int, default=5, help="calibration pairs (rest + MI)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(args)
    print(f"open http://127.0.0.1:{args.port}  (patient {args.patient}, replay x{args.speed}, k = {args.k})")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
