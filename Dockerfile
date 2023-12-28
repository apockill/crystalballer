FROM luxonis/depthai-library:v2.24.0.0

ARG POETRY_VERSION=1.5.1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org --output /tmp/install-poetry.py \
    && POETRY_HOME=/usr/local python3 /tmp/install-poetry.py --version "${POETRY_VERSION}"
RUN poetry config virtualenvs.create false
WORKDIR /app

COPY pyproject.toml poetry.lock .
RUN poetry install --no-dev
COPY . .
RUN poetry install --no-dev  # This last run will install the scripts for the app

