FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv fetches a standalone Python 3.13 build for the aviationstack MCP
# server's dedicated venv (aviationstack-mcp requires Python >=3.13,
# newer than this image's base Python). It also provides uvx, which
# mcp_client.py checks for at runtime.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Dedicated venv for the aviationstack MCP server (see mcp_client.py).
# mcp[cli] is pinned below 2.0 because aviationstack-mcp depends on the
# mcp.server.fastmcp module that the 2.0 release removed.
RUN uv python install 3.13 \
    && uv venv --python 3.13 .mcp_aviationstack_venv \
    && uv pip install --python .mcp_aviationstack_venv/bin/python \
        "mcp[cli]==1.28.1" aviationstack_mcp

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]