#!/bin/bash
set -euxo pipefail

cd face_tracker
poetry run isort crystalballer/ tests/
poetry run black crystalballer/ tests/
