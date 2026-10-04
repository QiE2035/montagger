#!/bin/sh
# Launch montagger from its own folder, so the config, models and database
# stay next to the plugin.
cd "$(dirname "$0")"
exec python3 -m montagger
