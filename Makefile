# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

accelerator=cuda
port=8001

test:
	uv run --extra dev --extra $(accelerator) pytest tests

update-precommit:
	uv run --extra dev --no-default-groups pre-commit autoupdate

style:
	uv run python tools/populate_readme.py
	uv run python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	uv run --extra dev --no-default-groups pre-commit run --all-files

website:
	$(MAKE) -C docs html
	sphinx-autobuild docs docs/_build/html --port $(port)
