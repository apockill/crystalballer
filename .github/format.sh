#!/bin/bash
set -euxo pipefail

poetry run isort crystalballer/ tests/
poetry run black crystalballer/ tests/
