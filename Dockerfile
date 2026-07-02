# DAS governance control plane — the deployable unit.
# Serves governance_api.py: routing-with-provenance, right-to-be-forgotten, and
# the tamper-evident audit log. NumPy + Flask only (no torch), so the image is
# small and the attack surface narrow.
FROM python:3.11-slim

WORKDIR /app

# Install the governance stack (numpy core + flask web extra) plus the platform
# extra (pyyaml), so the `das deploy` CLI can read a client.yaml spec at boot.
# Copying the package + pyproject first lets this layer cache across code changes.
COPY pyproject.toml README.md ./
COPY das/ ./das/
COPY das_torch.py das_text.py ./
# Default: the small torch-free governance image. Build with
#   --build-arg DAS_EXTRAS=web,platform,hf
# for the LoRA-on-MiniLM expert backend (`backend: lora-minilm` in the spec) —
# the image grows by the torch/sentence-transformers footprint, nothing else
# changes: same entrypoint, same guarantees, same API.
ARG DAS_EXTRAS=web,platform
RUN pip install --no-cache-dir ".[${DAS_EXTRAS}]"

COPY apps/governance_api.py ./
COPY deploy/entrypoint.sh /usr/local/bin/das-entrypoint
RUN chmod +x /usr/local/bin/das-entrypoint

# Persisted state lives on a volume; the audit secret is supplied at runtime and
# is NEVER baked into the image (the code default is dev-only — override it).
# Set DAS_SPEC (path to a mounted client.yaml) to have the container stand up the
# fleet from that spec via `das deploy` on first boot — the declarative front door.
ENV DAS_STATE=/data \
    DAS_PORT=5070
VOLUME ["/data"]
EXPOSE 5070

# Non-root, and make the state dir writable by it.
RUN useradd --create-home --uid 10001 das && mkdir -p /data && chown das:das /data
USER das

# Liveness: the API must report a verifying audit chain.
HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,json; \
u='http://localhost:%s/health'%os.environ.get('DAS_PORT','5070'); \
j=json.load(urllib.request.urlopen(u,timeout=3)); \
exit(0 if j.get('audit_chain_ok') else 1)"

# The entrypoint runs `das deploy $DAS_SPEC` first (if a spec is set and state is
# not yet materialized), then execs the CMD to serve the fleet.
ENTRYPOINT ["das-entrypoint"]

# Production WSGI server (F5). ONE worker keeps the single-writer audit chain
# intact (a single replica owns the chain); threads handle concurrency. The
# startup guards in governance_api still run at import, so a misconfigured prod
# deploy fails fast here too.
# `exec` so gunicorn replaces the shell and receives SIGTERM directly — clean
# shutdown instead of a 10s force-kill with a state write possibly in flight.
CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 8 --timeout 30 --bind 0.0.0.0:${DAS_PORT:-5070} governance_api:app"]
