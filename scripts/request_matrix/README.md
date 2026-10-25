# Request matrix

Runs generation requests across the combinations a client can send, on the real checkpoints, and reports what
breaks: crashes, replies that break a guarantee, valid requests that are refused, and invalid ones that are not.

A full product of the request options is millions of requests, so the matrix is a **covering array**: every
pair of values of any two options appears in some request, plus the combinations most likely to break in full
(remix and exact rhythm over several bars, the auto planner with dropped context and PROB, the mask modes with
AR, and so on). Bugs that need two options together, like remix over two fill-in bars, are caught in a few
hundred requests per model.

| File | |
|---|---|
| `axes.py` | the options and their values, the rules for a valid combination, the risky combinations, the probes that must be refused |
| `covering.py` | the covering array (greedy, deterministic for a seed) |
| `build.py` | picks GigaMIDI test pieces by kind (4/4 with and without drums, 3/4, 6/8, sparse, wide) and builds each request |
| `checks.py` | what a reply must satisfy: only the asked bars change, masks and remix are obeyed, bar lengths, no runaway generation |
| `run.py` | runs one model's cases (optionally a shard), shrinks failures to the fewest non-default options |
| `report.py` | groups failures by signature across shards |
| `cluster.sh` | one shard on an Alliance GPU node |

```bash
python scripts/request_matrix/run.py --data <GigaMIDI test split, or a folder of .mid> --model prism_medium \
    --out runs/matrix --limit 10            # a few cases
python scripts/request_matrix/report.py runs/matrix --md runs/matrix/report.md
```

Each failure is saved with its score and request in `<out>/failures/`, so it can be replayed with
`InferenceEngine.session(Score.from_dict(f["score"]), GenerationRequest.from_dict(f["request"])).run()`.
