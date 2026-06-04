# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

update-precommit:
	pre-commit autoupdate

style:
	python copyright.py --repo ./ --header "Copyright (c) $$(date +%Y), __authors__" --extra-name "Mayank Mishra"
	pre-commit run --all-files
