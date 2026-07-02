#!/bin/sh
# DAS container front door.
#
# If DAS_SPEC names a client spec (client.yaml / client.json) and the state
# directory has not been materialized yet, stand up the fleet FROM THE SPEC with
# `das deploy` — the declarative spec is how the container boots. Then hand off
# to the server (the CMD), which loads that persisted state and serves it.
#
# Behaviour:
#   * DAS_SPEC set, no state yet   -> deploy the spec, persist to DAS_STATE, serve.
#   * DAS_SPEC set, state present  -> skip deploy (restart-safe / idempotent), serve.
#   * DAS_SPEC unset               -> unchanged: the API bootstraps a demo fleet.
#
# The audit secret is shared by both steps via $DAS_AUDIT_SECRET (or the spec's
# audit.secret_file), so the log `das deploy` signs is loadable by the server.
set -e

STATE="${DAS_STATE:-/data}"

# Invoke the CLI via `python -m` so it works regardless of whether the `das`
# console script is on PATH (it is in the image; this is just belt-and-braces).
DAS_CLI="python -m das.platform.cli"

if [ -n "$DAS_SPEC" ] && [ ! -f "$STATE/control_plane.json" ]; then
  echo "das: bootstrapping fleet from '$DAS_SPEC' -> '$STATE'"
  $DAS_CLI deploy "$DAS_SPEC" --save "$STATE"
elif [ -n "$DAS_SPEC" ]; then
  echo "das: state already present in '$STATE' — skipping deploy, serving existing fleet"
fi

exec "$@"
