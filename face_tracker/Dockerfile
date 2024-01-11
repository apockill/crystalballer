FROM luxonis/depthai-library:v2.24.0.0

ARG POETRY_VERSION=1.7.0

# Open3d needs this to be able to work
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# Install Poetry
RUN curl -sSL https://install.python-poetry.org --output /tmp/install-poetry.py \
    && POETRY_HOME=/usr/local python3 /tmp/install-poetry.py --version "${POETRY_VERSION}"
RUN poetry config virtualenvs.create false
WORKDIR /app

# Copy in dependencies and install them
COPY pyproject.toml poetry.lock .
RUN poetry install --no-dev

# Copy in the rest of the code and run installation again, so that binaries for scripts
# are generated and put in the path
COPY . .
RUN poetry install --no-dev

