# DAS governance control plane — the deployable unit.
# Serves governance_api.py: routing-with-provenance, right-to-be-forgotten, and
# the tamper-evident audit log. NumPy + Flask only (no torch), so the image is
# small and the attack surface narrow.
FROM python:3.11-slim

WORKDIR /app

# Install just the governance stack (numpy core + flask web extra). Copying the
# package + pyproject first lets this layer cache across code-only changes.
COPY pyproject.toml README.md ./
COPY das/ ./das/
COPY das_torch.py ./
RUN pip install --no-cache-dir .[web]

COPY governance_api.py ./

# Persisted state lives on a volume; the audit secret is supplied at runtime and
# is NEVER baked into the image (the code default is dev-only — override it).
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

CMD ["python", "governance_api.py"]
