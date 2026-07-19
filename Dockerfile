# --- Base image ---
# Start from an official, minimal Python image rather than a full OS image.
# "slim" has just enough to run Python — smaller image, faster builds/pulls,
# smaller attack surface than a full Ubuntu base.
FROM python:3.10-slim

# --- Working directory ---
# All subsequent commands (COPY, RUN, CMD) happen relative to this path
# inside the container. Doesn't need to match anything on your machine.
WORKDIR /code

# --- Install dependencies FIRST, before copying the rest of the code ---
# This ordering matters for build speed: Docker caches each step. If you
# copied all your code first and installed dependencies after, ANY code
# change (even editing a comment) would force a full dependency reinstall
# on every rebuild. By copying only requirements.txt first, Docker can
# reuse the cached "pip install" layer as long as requirements.txt itself
# hasn't changed — so rebuilds after a code change are much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Now copy the actual application code ---
COPY app/ ./app/
COPY models/ ./models/

# --- Document which port the app listens on ---
# This is informational (doesn't actually publish the port) — the real
# port mapping happens in `docker run -p` at runtime.
EXPOSE 8000

# --- The command that runs when the container starts ---
# Note: no --reload here (that's a dev-only feature that watches files for
# changes — irrelevant and wasteful in a built, immutable container).
# --host 0.0.0.0 is required so the server accepts connections from outside
# the container, not just from localhost within it.
#
# ${PORT:-8000} uses the $PORT environment variable if the hosting platform
# sets one (Render, and most PaaS platforms, assign a port dynamically at
# runtime and expect the app to listen on it) — falls back to 8000 for
# local `docker run` where no $PORT is set.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]