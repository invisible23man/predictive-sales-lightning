up:
	docker compose up --build

down:
	docker compose down

train:
	poetry run python -m src.ml.train.train

lint:
	poetry run pre-commit run --all-files

test:
	poetry run pytest

dashboard:
	streamlit run src/ui/streamlit_app/app.py

api:
	poetry run uvicorn src.app.main:app --host 0.0.0.0 --port 8000
