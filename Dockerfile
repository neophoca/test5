FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY demo/requirements.txt /app/demo/requirements.txt
RUN pip install --no-cache-dir -r /app/demo/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "8000"]