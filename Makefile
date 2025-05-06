up:
\tdocker-compose up --build

train:
\tpoetry run python -m src.ml.train.train

lint:
\tpoetry run pre-commit run --all-files

test:
\tpoetry run pytest
