# Source2Agent deployment

This container exposes the source-to-artifact path that is currently proven by the repository:

- `GET /health`
- `POST /compile`
- `POST /reference-eval`

The reference endpoint is explicitly symbolic. It is a contract check, not a trained model and not evidence of generalization.

## Run locally

```bash
docker build -t source2agent:local .
docker run --rm -p 8000:8000 source2agent:local
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health

python - <<'PY' | curl -sS \
  -H 'Content-Type: application/json' \
  --data-binary @- \
  http://127.0.0.1:8000/compile
import json
from pathlib import Path
print(json.dumps({
    "volume_id": "lantern-ledger-demo",
    "units": json.loads(Path("examples/lantern_ledger/units.json").read_text()),
    "seed": 7,
}))
PY
```

## Hosted deployment contract

A hosted deployment is not complete until the following are recorded:

1. image or source revision;
2. endpoint URL;
3. health response;
4. compile response for the Lantern fixture;
5. reference-evaluation response;
6. model/evidence mode;
7. cost and runtime;
8. rollback target.

Prime Lab deployment of a neural adapter is a separate milestone. It requires a pinned taskset, harness, runtime, reward contract, model artifact, and an inference endpoint. Do not replace those with a README claim.
