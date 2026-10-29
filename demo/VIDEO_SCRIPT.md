# HemiShift: 2-minute demo video script

Start the demo before recording, with the browser at http://127.0.0.1:8000 and the session not
yet started:

```
.venv/Scripts/python -I demo/server.py --patient 22 --speed 4
```

At ×4, calibration takes about 40 s and the first live decisions follow right after. That is
enough for one recording.

| Time | On screen | Voice-over (about 280 words) |
|---|---|---|
| 0:00–0:15 | Title slide or the dashboard header | "After a stroke, brain–computer interfaces can help patients recover arm movement: the computer detects the attempt to move and gives feedback. But every patient needs a calibration session first, and stroke patients tire quickly." |
| 0:15–0:35 | README results table | "So the question is: can we borrow data to shorten calibration? We tested four sources on two public stroke cohorts: healthy people's EEG, simulated stroke EEG, a large EEG foundation model, and other stroke patients." |
| 0:35–0:50 | Press **Start session**; EEG traces start scrolling | "This is a real patient from our locked test set, never seen during development. Their raw 64-channel recording is streamed live over Lab Streaming Layer, the same protocol real EEG headsets use." |
| 0:50–1:10 | The calibration steps fill; then the gate box appears | "The system calibrates on just five rest and five imagery trials. Then a safety gate checks, from calibration data alone, whether the decoder can be trusted, instead of giving the patient misleading feedback." |
| 1:10–1:30 | Live accuracy cards and the green/red history strip | "From now on, every trial is decoded about 20 milliseconds after its window ends. On the left, the patient's own decoder. On the right, HemiShift: the same calibration plus data from 16 other stroke patients, aligned geometrically." |
| 1:30–1:50 | Figure 1 from the paper (locked test) | "In a pre-registered test on never-seen patients, this lifted accuracy from 77 to 81 percent with five calibration trials, as good as the patient's own decoder with ten. Healthy data and simulated stroke data did not help at all, because stroke brains organise movement differently." |
| 1:50–2:00 | Repository page | "Everything is open: the benchmark code, the pre-registration history and this real-time system." |

**Accuracy caveats for the narration:**
- The 77 → 81 % figure is the mean over the 8 locked-test patients (one-sided p = 0.035).
- The single patient on screen can do better or worse than that.
- The replay runs at ×4. Say "about 20 milliseconds of processing", not "real time at ×4".
