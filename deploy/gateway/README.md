# Reference authn gateway (PLATFORM_PLAN §10.1)

TLS + OIDC in front of the governance API, wiring the `DAS_TRUSTED_PROXY_SECRET`
contract end-to-end. The API's own hardening (`DAS_ENV=production`) refuses to
start without real secrets and rejects any request that did not transit this
gateway; the gateway is what performs the actual user authentication and is the
ONLY thing allowed to set `X-DAS-Actor` (SECURITY_REVIEW F2).

```
client ──TLS──> nginx ──auth_request──> oauth2-proxy ──OIDC──> your IdP
                  │ authenticated requests only
                  └─> das-governance   X-DAS-Actor: <verified email>
                                       X-DAS-Proxy-Auth: <shared secret>
```

## Bring-up

1. **Secrets** (never commit these):
   ```sh
   mkdir -p secrets tls
   openssl rand -hex 32 > secrets/das_audit_secret
   openssl rand -hex 32 > secrets/das_proxy_secret
   echo '<client secret from your IdP>' > secrets/oidc_client_secret
   openssl rand -base64 32 | tr -- '+/' '-_' > secrets/cookie_secret
   ```
2. **TLS** — real certs in `tls/tls.crt` / `tls.key`, or a self-signed dev pair:
   ```sh
   openssl req -x509 -newkey rsa:2048 -nodes -days 30 \
     -keyout tls/tls.key -out tls/tls.crt -subj "/CN=localhost"
   ```
3. **IdP** — register a confidential client `das-governance` with redirect URL
   `https://<host>/oauth2/callback`, then set `--oidc-issuer-url` (and tighten
   `--email-domain`) in `docker-compose.yaml`.
4. **Fleet** — put your `client.yaml` next to the compose file. Its `users`
   must name the OIDC **emails** (the gateway maps the verified email to
   `X-DAS-Actor`, and RBAC is enforced on that name).
5. `docker compose up` — then `https://<host>/experts` should bounce you
   through your IdP and come back tenant-scoped.

## What this proves / what it doesn't

* Proves: only authenticated principals reach the API; the API only honours
  `X-DAS-Actor` on requests carrying the gateway's secret; identity headers
  from clients are overwritten unconditionally at the gateway.
* Does NOT cover: authorization beyond DAS's own RBAC, IdP hardening, or
  network policy between the containers (add a private network / mTLS for
  zero-trust environments). Rate limits belong here at the proxy for full
  F5 coverage.
* The freshness anchor volume (`das-anchor`) exists so a rolled-back state
  snapshot is detected at load (F1) — in production bind it to storage the
  state writer cannot modify, and back it up separately (docs/RUNBOOK.md).

Kubernetes: `deploy/k8s.yaml` carries the same contract (file-mounted secrets,
production guards, and an Ingress stub for cert-manager + oauth2-proxy
annotations).
