"""Entry point: `python3 -m montagger`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, logx
from .app import AppContext, create_app
from .config import Provider
from .engine.backend import Engine
from .queue import Runner
from .store import Store


def repo_root() -> Path:
    # The checkout root: montagger/ sits directly under it.
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="montagger", description=__doc__)
    parser.add_argument(
        "--config", default=str(repo_root() / "montagger.toml"), help="config file path"
    )
    parser.add_argument(
        "--bind", default="", help="override server.bind_address (host:port)"
    )
    parser.add_argument(
        "--version", action="version", version=f"montagger {__version__}"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser()
    cfg = Provider(config_path)
    if not config_path.exists():
        cfg.update(lambda _c: None)  # persist defaults so the operator can edit them
        print(f"montagger: wrote default config to {config_path}")

    logx.setup(cfg.current().log.debug)
    log = logx.get("main")

    if args.bind:
        cfg.update(lambda c: setattr(c.server, "bind_address", args.bind))

    import uvicorn

    try:
        engine = Engine(cfg, repo_root())
    except Exception as err:
        # A provider or model problem is fatal and explained, never silently
        # downgraded.
        print(f"montagger: refusing to start: {err}", file=sys.stderr)
        return 1
    log.info(
        "provider %s (%s) on %s",
        engine.provider_short,
        engine.provider,
        cfg.current().model_dir(repo_root()),
    )

    store_ = Store(repo_root() / "montagger.sqlite")
    queue_cfg = cfg.current().queue
    runner = Runner(
        engine,
        store_,
        queue_cfg.max_pending,
        keep_recent=queue_cfg.keep_recent,
        max_job_objects=queue_cfg.max_job_objects,
    )

    # The pairing card and relay endpoints are part of the app; the offer
    # loop stays idle until a monbooru api_url is configured.
    from .pair import Integration

    integration = Integration(cfg, repo_root() / "credentials.json", runner)
    integration.start()

    app = create_app(AppContext(cfg, engine, store_, runner, integration))

    host, _, port = cfg.current().server.bind_address.rpartition(":")
    log.info("listening on http://%s:%s", host or "0.0.0.0", port or "8457")
    uvicorn.run(
        app, host=host or "0.0.0.0", port=int(port or 8457), log_level="warning"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
