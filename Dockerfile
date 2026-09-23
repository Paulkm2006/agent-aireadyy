FROM docker.m.daocloud.io/python:3.13-slim

WORKDIR /app
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    docker.io \
    git \
    openjdk-21-jre-headless \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY README.md ./

COPY src/ src/
COPY profiles/ profiles/
COPY scripts/ scripts/

# Install every declared runtime extra used by the web service and dataset
# construction workers.  Keeping this in one project install ensures the
# Docker image follows pyproject.toml rather than an unrelated PyPI package.
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple --no-cache-dir -e ".[agents-sdk,dev,web,dataset-construction,dataset-construction-ortools-worker]"

RUN mkdir -p /app/data /app/runs /app/.agent_cache

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agent.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
