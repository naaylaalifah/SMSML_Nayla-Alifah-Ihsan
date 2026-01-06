from flask import Flask, request, jsonify, Response
import requests
import time
import psutil

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# Total request inference
REQUEST_TOTAL = Counter(
    "request_total",
    "Total request ke model ML"
)

# Total error inference
ERROR_TOTAL = Counter(
    "error_total",
    "Total error saat inference model"
)

# Latency inference
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latency inference model (detik)",
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10)
)

# CPU & Memory
CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU usage dalam persen"
)

MEMORY_USAGE = Gauge(
    "memory_usage_percent",
    "Memory usage dalam persen"
)

@app.route("/metrics")
def metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_TOTAL.inc()

    MODEL_API_URL = "http://model:5005/invocations"  # SESUAI DOCKER
    headers = {"Content-Type": "application/json"}
    payload = request.get_json()

    try:
        response = requests.post(
            MODEL_API_URL,
            json=payload,
            headers=headers,
            timeout=10
        )

        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)

        if response.status_code != 200:
            ERROR_TOTAL.inc()
            return jsonify({"error": "Model inference failed"}), 500

        return jsonify(response.json()), 200

    except Exception as e:
        ERROR_TOTAL.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
