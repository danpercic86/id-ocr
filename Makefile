setup:
	poetry config virtualenvs.in-project true
	poetry install
	poetry shell

lint:
	black .
