"""
Cron-friendly automation worker for the DAS growing-child loop.

This script talks to a running governance API instead of importing
governance_api.py directly. That keeps automation pointed at the same live state,
audit secret, trusted-proxy contract, and persistence settings as production.

Example:
    python apps/growth_worker.py --actor root --max-attempts 4 --save
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def _request(base, path, actor, proxy_secret=None, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"X-DAS-Actor": actor}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if proxy_secret:
        headers["X-DAS-Proxy-Auth"] = proxy_secret
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        raise SystemExit(f"{path} failed with HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"could not reach {base}: {e.reason}") from e


def main(argv=None):
    p = argparse.ArgumentParser(description="Run an automated DAS growth cycle.")
    p.add_argument("--base", default="http://127.0.0.1:5070",
                   help="governance API base URL")
    p.add_argument("--actor", default="root",
                   help="actor sent as X-DAS-Actor")
    p.add_argument("--proxy-secret",
                   help="optional X-DAS-Proxy-Auth credential")
    p.add_argument("--teacher", action="append", dest="teachers",
                   help="teacher id to use; repeat for multiple teachers")
    p.add_argument("--register-teacher",
                   help="register or replace a dynamic teacher before running")
    p.add_argument("--teacher-provider", default="local-vector",
                   help="provider for --register-teacher: local-vector, openai-compatible, ollama, custom-json")
    p.add_argument("--teacher-label",
                   help="display label for --register-teacher")
    p.add_argument("--teacher-endpoint",
                   help="endpoint/base URL for an LLM teacher")
    p.add_argument("--teacher-model",
                   help="model name for an LLM teacher")
    p.add_argument("--teacher-api-key",
                   help="optional API key for an LLM teacher")
    p.add_argument("--teacher-max-examples", type=int, default=48,
                   help="max examples requested per LLM lesson call")
    p.add_argument("--teacher-temperature", type=float, default=0.2,
                   help="LLM sampling temperature")
    p.add_argument("--create-expert", action="append", dest="create_experts",
                   help="create a new expert before the cycle; repeat for several")
    p.add_argument("--tenant", default="learning",
                   help="tenant for created experts")
    p.add_argument("--specialty",
                   help="specialty branch for created experts")
    p.add_argument("--parent",
                   help="parent branch label for created experts")
    p.add_argument("--create-steps", type=int, default=180,
                   help="seed training steps for created experts")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="maximum experts to try this cycle")
    p.add_argument("--steps", type=int, default=120,
                   help="candidate training steps per expert")
    p.add_argument("--n-train", type=int, default=180,
                   help="teacher training examples per expert")
    p.add_argument("--n-eval", type=int, default=120,
                   help="teacher evaluation examples per expert")
    p.add_argument("--lr", type=float, default=0.05,
                   help="candidate training learning rate")
    p.add_argument("--save", action="store_true",
                   help="call POST /save after the cycle")
    p.add_argument("--sync-mobile-models", action="store_true",
                   help="call POST /growth/mobile/save after create/cycle work")
    p.add_argument("--no-cycle", action="store_true",
                   help="only create requested experts; skip the automated cycle")
    args = p.parse_args(argv)

    output = {}
    if args.register_teacher:
        teacher_body = {
            "id": args.register_teacher,
            "provider": args.teacher_provider,
            "label": args.teacher_label or args.register_teacher,
            "endpoint": args.teacher_endpoint,
            "model": args.teacher_model,
            "api_key": args.teacher_api_key,
            "max_examples": args.teacher_max_examples,
            "temperature": args.teacher_temperature,
            "replace": True,
        }
        output["registered_teacher"] = _request(
            args.base, "/growth/teachers", args.actor, args.proxy_secret, teacher_body
        )
        if not args.teachers:
            args.teachers = [args.register_teacher]

    if args.create_experts:
        created = []
        for name in args.create_experts:
            create_body = {
                "name": name,
                "tenant": args.tenant,
                "specialty": args.specialty,
                "parent": args.parent,
                "teacher": args.teachers[0] if args.teachers else "qwen-8b-teacher",
                "steps": args.create_steps,
                "n_train": args.n_train,
                "n_eval": args.n_eval,
                "lr": args.lr,
            }
            created.append(_request(
                args.base, "/growth/create_expert", args.actor, args.proxy_secret, create_body
            ))
        output["created"] = created

    body = {
        "max_attempts": args.max_attempts,
        "steps": args.steps,
        "lr": args.lr,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
    }
    if args.teachers:
        body["teachers"] = args.teachers

    result = output
    if not args.no_cycle:
        result.update(_request(args.base, "/growth/auto/run", args.actor, args.proxy_secret, body))
    if args.sync_mobile_models:
        result["mobile_models"] = _request(
            args.base, "/growth/mobile/save", args.actor, args.proxy_secret, {}
        )
    if args.save:
        result["save"] = _request(args.base, "/save", args.actor, args.proxy_secret, {})

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
