setup:
	poetry config virtualenvs.in-project true
	poetry shell
	poetry install

lint:
	black .
