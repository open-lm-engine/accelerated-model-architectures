# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

accelerator=cuda
port=8001
num_accelerators ?= $(shell uv run python -c "import torch; n=torch.cuda.device_count(); print(n if n > 0 else 1)" 2>/dev/null || echo 1)

test:
	uv run --extra dev --extra $(accelerator) pytest -n $(num_accelerators) tests

update-precommit:
	uv run --extra dev --no-default-groups pre-commit autoupdate

style:
	uv run python tools/populate_readme.py
	uv run python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) $$(date +%Y), __authors__"
	uv run --extra dev --no-default-groups pre-commit run --all-files

website:
	uv run --extra dev python tools/build_docs.py

host-website:
	uv run --extra dev sphinx-autobuild docs docs/_build/html --port $(port)
