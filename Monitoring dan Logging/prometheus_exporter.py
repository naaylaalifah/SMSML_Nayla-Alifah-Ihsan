from prometheus_client import start_http_server, Counter, Gauge
import time
import random

# METRICS
inference_requests = Counter(
    "inference_requests_total",
    "Total inference requests"
)

inference_latency = Gauge(
    "inference_latency_seconds",
    "Inference latency in seconds"
)

model_accuracy = Gauge(
    "model_accuracy",
    "Model accuracy"
)

if __name__ == "__main__":
    start_http_server(8000)
    while True:
        inference_requests.inc()
        inference_latency.set(random.uniform(0.1, 1.5))
        model_accuracy.set(0.87)
        time.sleep(5)