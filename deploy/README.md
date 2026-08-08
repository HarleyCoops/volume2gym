# Source2Agent deployment

This container exposes the source-to-artifact path that is currently proven by the repository:

- `GET /health`
- `GET /volumes`
- `GET /volumes/railroad-1959-v0`
- `POST /volumes/railroad-1959-v0/compile`
- `POST /volumes/railroad-1959-v0/reference-eval`
- `POST /compile`
- `POST /reference-eval`

The reference endpoint is explicitly symbolic. It is a contract check, not a
trained model and not evidence of generalization. The container defaults to
`named-volume-only` mode: `/compile` and `/reference-eval` return `403` unless
the operator explicitly sets `SOURCE2AGENT_ALLOW_CUSTOM_INPUT=1`.

## Run locally

```bash
docker build -t source2agent:local .
docker run --rm -p 8000:8000 source2agent:local
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/volumes/railroad-1959-v0
curl -X POST http://127.0.0.1:8000/volumes/railroad-1959-v0/compile
curl -X POST http://127.0.0.1:8000/volumes/railroad-1959-v0/reference-eval

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

To run that custom-input example in the container, start it with the explicit
opt-in:

```bash
docker run --rm -p 8000:8000 \
  -e SOURCE2AGENT_ALLOW_CUSTOM_INPUT=1 \
  source2agent:local
```

Requests are capped at 256 KiB and custom payloads at 100 units by default.
Only one uncached compile or evaluation runs at a time, and named deterministic
results are cached in process. This remains a low-capacity evidence API without
authentication or provider-level rate limiting, not a hardened multi-tenant
service.

The named compile endpoint rebuilds the checked volume from the committed rule
corpus. The named reference endpoint evaluates only the held-out test split.
Neither endpoint runs a neural model.

On configured branches and version tags, GitHub Actions publishes the container
to `ghcr.io/harleycoops/volume2gym`. A container publication is not an always-on
HTTP host; record the provider URL and health evidence separately when one is
provisioned.

## Always-on Railway deployment

[`railway.json`](../railway.json) selects the existing Dockerfile, gates cutover
on `GET /health`, and restarts the process after exits. Railway services are
long-running by default, but the operator must verify **Serverless is disabled**
in the service's deploy settings; Railway does not currently expose that toggle
through config-as-code. The health check is a deployment-readiness gate, not
continuous liveness monitoring.

From an authenticated Railway CLI linked to the intended project and service:

```bash
npm install -g @railway/cli
railway login --browserless
railway link
railway up --ci
railway domain
```

For automation, use a Railway project token through the host or CI secret store;
never commit `RAILWAY_TOKEN`, account tokens, project IDs, or service credentials.
The Dockerfile already listens on Railway's injected `PORT`, so no start-command
override is required.

After the CLI prints the public domain, verify the exact deployed revision and
all four named-volume contracts, then store the emitted JSON in the deployment
evidence directory:

```bash
python scripts/smoke_hosted_deployment.py \
  https://YOUR-SERVICE.up.railway.app \
  --expected-revision "$(git rev-parse HEAD)" \
  --output evidence/deployments/source2agent-api-railway.json
```

The script fails on a revision mismatch or any change to the expected health,
descriptor, 457-unit/2,742-task compile, structural split counts, or 276-task
symbolic reference result. The generated evidence does not claim neural
inference or learning.

## Hosted deployment contract

A hosted deployment is not complete until the following are recorded:

1. image or source revision;
2. endpoint URL;
3. health response;
4. named railroad descriptor and compile response;
5. named held-out symbolic reference-evaluation response;
6. model/evidence mode;
7. cost and runtime;
8. rollback target.

Rollback uses Railway's last known-good deployment/image, followed by a Git
revert or fix-forward so the source branch and running revision converge.

Prime Lab deployment of a neural adapter is a separate milestone. It requires a pinned taskset, harness, runtime, reward contract, model artifact, and an inference endpoint. Do not replace those with a README claim.
