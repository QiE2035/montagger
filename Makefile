.PHONY: run ui ui-dev ui-install test lint clean

VERSION := $(shell cat VERSION.md 2>/dev/null | tr -d '[:space:]')

# Run the backend only. The web UI is served from web/dist when present;
# without a build the API and /docs still answer.
run:
	python3 -m montagger

# Frontend: everything goes through pnpm, never npm. Dependency installs use
# pnpm's default store location; nothing is redirected.
ui-install:
	cd web && pnpm install

ui-dev:
	cd web && pnpm dev

ui:
	cd web && pnpm build

test:
	python3 -m pytest

lint:
	python3 -m compileall -q montagger

clean:
	rm -rf web/dist web/node_modules .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
