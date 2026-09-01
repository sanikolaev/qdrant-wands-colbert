FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY demo.py web_demo.py ./

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "web_demo.py", "--host", "0.0.0.0", "--port", "8000"]
