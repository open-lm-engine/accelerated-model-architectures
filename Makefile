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
	uv run --extra dev $(MAKE) -C docs clean
	uv run --extra dev $(MAKE) -C docs html
	uv run --extra dev sphinx-apidoc -e -o docs . tests
	uv run --extra dev python tools/clean_rst_headings.py
	uv run --extra dev sphinx-autobuild docs docs/_build/html --port $(port)
