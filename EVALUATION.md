# Scenario smoke evaluation

Local CPU ONNX run, 2026-09-21. Fixed presets, every third source frame; no training.
These few clips are regression examples, not an accuracy benchmark.

| Preset | Processed frames | Candidate events |
| --- | ---: | --- |
| pedestrians / observe | 265 | 0 |
| normal traffic | 180 | 0 |
| reversed traffic (controlled test) | 67 | 3 wrong-direction candidates |
| staged slap | 49 | 0 (missed) |
| staged fight | 68 | 1 interaction candidate |
| nonviolent interaction | 53 | 1 interaction candidate (false candidate) |

Live Groq checks with the configured `qwen/qwen3.8-27b` model:

- Reversed traffic: `confirmed`, warning, operator notification.
- Nonviolent interaction: `not_confirmed`, info, no action.
- Staged fight: candidate at 2.2 seconds, frames at 0.0/1.0/2.2 seconds;
  `not_confirmed`, info, no action. The VLM missed the labelled positive clip.

The nonviolent live check preceded the final increase in candidate duration;
its candidate timing differs from the final offline run. Provider outputs are
nondeterministic. Initial attempts hit HTTP 429; they retained local informational
reports. Contact-sheet payloads and a 70-second process throttle reduced request
pressure. Account-wide quotas still apply.

The experiment fixes headcount-based urgency but does **not** establish reliable
fight detection. Motion selection and sparse early frames can miss short actions.
Improvement requires labelled temporal windows, post-trigger evidence, a trained
action model, and held-out cameras/actors. Never equate candidate count with true
violations. The wrong-direction positive is artificially reversed footage.

Reproduce with `python -m src.evaluate_scenarios` and explicitly opt into paid or
quota-consuming calls with `python -m src.check_scenarios_vlm`.
Unit/integration validation: 21 tests passed; one upstream deprecation warning.
Docker, GPU providers and live street cameras were not tested in this run.
