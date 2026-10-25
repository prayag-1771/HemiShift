"""Lab Streaming Layer (LSL) transport for the live demo: replay a recording, decode the stream.

LSL is how real EEG amplifiers publish data, so the receiver below would run unchanged on a
live headset. The replay publishes two streams, as an amplifier plus a stimulus program would:
  <name>-EEG      every channel of the recording, microvolts, at its own sampling rate
  <name>-Markers  one string per cue ('rest', 'affected_mi', 'sixth_finger_mi')
Samples are time-stamped in recording time (start + sample / sfreq), so markers line up with
the data at any replay speed.
"""

import threading
import time

import numpy as np

from hemishift.online.session import CausalBandpass, covariances, harmonised_epoch, window_samples

DISPLAY = ("FC3", "C3", "CP3", "Cz", "FC4", "C4", "CP4")  # channels sent to the dashboard


def wang2026sf_recording(subject, phases=("pre",)):
    """Raw Tianjin runs back to back: (data µV (n_channels, n_samples), names, sfreq, events)
    with events as (sample, label) in recording order."""
    import mne

    from hemishift.data.loaders import WANG2026SF_EVENTS
    from hemishift.paths import RAW_DIR

    raws = []
    for phase in phases:
        runs = sorted((RAW_DIR / "wang2026sf" / "sourcedata" / f"sub-{subject:02d}" / phase)
                      .glob("*_original.set"))
        raws += [mne.io.read_raw_eeglab(p, preload=True, uint16_codec="latin1", verbose="error")
                 for p in runs]
    raw = mne.concatenate_raws(raws, verbose="error")
    events, event_id = mne.events_from_annotations(raw, verbose="error")
    names = {code: WANG2026SF_EVENTS[key] for key, code in event_id.items() if key in WANG2026SF_EVENTS}
    marked = [(int(sample), names[code]) for sample, _, code in events if code in names]
    return raw.get_data() * 1e6, list(raw.ch_names), float(raw.info["sfreq"]), marked


class Replay(threading.Thread):
    """Publish a recording on LSL in real time (times `speed`)."""

    def __init__(self, data, ch_names, sfreq, events, speed=1.0, name="HemiShift", chunk_s=0.0625):
        super().__init__(daemon=True)
        self.data, self.ch_names, self.sfreq, self.events = data, ch_names, sfreq, events
        self.speed, self.name, self.chunk = speed, name, max(1, int(round(chunk_s * sfreq)))
        self.stop_flag = threading.Event()

    def run(self):
        from pylsl import StreamInfo, StreamOutlet, local_clock

        info = StreamInfo(f"{self.name}-EEG", "EEG", len(self.ch_names), self.sfreq, "float32",
                          f"{self.name}-eeg")
        channels = info.desc().append_child("channels")
        for ch in self.ch_names:
            channels.append_child("channel").append_child_value("label", ch).append_child_value("unit", "microvolts")
        eeg = StreamOutlet(info, chunk_size=self.chunk)
        markers = StreamOutlet(StreamInfo(f"{self.name}-Markers", "Markers", 1, 0, "string",
                                          f"{self.name}-markers"))
        eeg.wait_for_consumers(timeout=30)
        markers.wait_for_consumers(timeout=30)
        t0, start = local_clock(), time.perf_counter()
        n, i, next_event = self.data.shape[1], 0, 0
        while i < n and not self.stop_flag.is_set():
            j = min(i + self.chunk, n)
            delay = start + (j / self.sfreq) / self.speed - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            while next_event < len(self.events) and self.events[next_event][0] < j:
                sample, label = self.events[next_event]
                markers.push_sample([label], t0 + sample / self.sfreq)
                next_event += 1
            eeg.push_chunk(np.ascontiguousarray(self.data[:, i:j].T, dtype=np.float32),
                           t0 + (j - 1) / self.sfreq)
            i = j


class StreamBuffer:
    """Filtered samples with their time stamps, oldest first; keeps the last `keep` samples."""

    def __init__(self, n_channels, keep):
        self.keep = keep
        self.data = np.empty((2 * keep, n_channels))
        self.stamps = np.empty(2 * keep)
        self.count = 0

    def extend(self, chunk, stamps):
        """Append a (n_samples, n_channels) chunk."""
        m = len(stamps)
        if self.count + m > len(self.stamps):  # drop the oldest samples beyond `keep`
            drop = self.count + m - self.keep
            self.data[:self.count - drop] = self.data[drop:self.count]
            self.stamps[:self.count - drop] = self.stamps[drop:self.count]
            self.count -= drop
        self.data[self.count:self.count + m] = chunk
        self.stamps[self.count:self.count + m] = stamps
        self.count += m

    def window(self, start_time, length):
        """`length` samples from the first one stamped at or after `start_time`, as
        (n_channels, length); None if they have not all arrived."""
        first = int(np.searchsorted(self.stamps[:self.count], start_time))
        if first + length > self.count:
            return None
        return self.data[first:first + length].T


class Receiver(threading.Thread):
    """Decode an LSL EEG + marker stream with a `Session`; report through `on_event(dict)`."""

    def __init__(self, session, affected_side, on_event, name="HemiShift", buffer_s=30.0):
        super().__init__(daemon=True)
        self.session, self.affected_side, self.on_event = session, affected_side, on_event
        self.name, self.buffer_s = name, buffer_s
        self.stop_flag = threading.Event()

    def run(self):
        from pylsl import StreamInlet, resolve_byprop

        eeg = StreamInlet(resolve_byprop("name", f"{self.name}-EEG", timeout=30)[0], max_buflen=60)
        markers = StreamInlet(resolve_byprop("name", f"{self.name}-Markers", timeout=30)[0])
        info = eeg.info()
        sfreq = info.nominal_srate()
        ch = info.desc().child("channels").child("channel")
        names = []
        for _ in range(info.channel_count()):
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        display = [names.index(c) for c in DISPLAY]
        offset, length = window_samples(sfreq)
        band = CausalBandpass(sfreq)
        buffer = StreamBuffer(len(names), int(self.buffer_s * sfreq))
        pending, trial, step = [], 0, max(1, int(round(sfreq / 50)))  # dashboard at ~50 Hz
        self.on_event({"type": "stream", "sfreq": sfreq, "channels": len(names), "display": list(DISPLAY)})

        while not self.stop_flag.is_set():
            samples, ts = eeg.pull_chunk(timeout=0.05)
            if ts:
                filtered = band(np.asarray(samples, dtype=float).T)
                buffer.extend(filtered.T, ts)
                self.on_event({"type": "eeg", "t": ts[-1],
                               "values": np.round(filtered[display][:, ::step], 2).tolist()})
            labels, mts = markers.pull_chunk(timeout=0.0)
            for (label,), t in zip(labels, mts):
                trial += 1
                pending.append((t, label, trial))
                self.on_event({"type": "cue", "trial": trial, "label": label, "t": t})
            while pending:
                t, label, n_trial = pending[0]
                if label == "sixth_finger_mi":  # not decoded; only resolves the previous rest
                    events = self.session.add(None, label, n_trial)
                else:
                    window = buffer.window(t + (offset - 0.5) / sfreq, length)
                    if window is None:
                        break  # this trial's window has not fully arrived yet
                    started = time.perf_counter()
                    C = covariances(harmonised_epoch(window, names, sfreq, self.affected_side)[None])[0]
                    events = self.session.add(C, label, n_trial)
                    for event in events:
                        event["latency_ms"] = round(1000 * (time.perf_counter() - started), 1)
                pending.pop(0)
                for event in events:
                    self.on_event(event)
