#!/bin/bash
set -euxo pipefail


# Run face_tracker specific linting
cd face_tracker
poetry run cruft check
poetry run mypy --ignore-missing-imports crystalballer/ tests/
poetry run isort --check --diff crystalballer/ tests/
poetry run black --check crystalballer/ tests/
poetry run flake8 crystalballer/ tests/ --darglint-ignore-regex '^test_.*'
poetry run bandit -r --severity medium high crystalballer/ tests/
poetry run vulture --min-confidence 100 crystalballer/ tests/
echo "Lint successful!"