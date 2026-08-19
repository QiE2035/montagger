"""montagger CLI: resolve the config file first, then serve."""

from __future__ import annotations

import logging
import sys

from montagger.settings import Settings, resolve_config_path


def main() -> None:
    # Resolve the TOML path before Settings is built (its source needs it).
    config_path = resolve_config_path()

    settings = Settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("montagger").info("config file: %s", config_path)

    import uvicorn

    from montagger.web.app import create_app

    host, _, port = settings.addr.partition(":")
    port = int(port or 8301)
    app = create_app(settings)
    if "--help" in sys.argv or "-h" in sys.argv:
        uvicorn.run(app, host=host, port=port, log_level="info")
        return
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()