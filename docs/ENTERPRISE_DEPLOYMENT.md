# AI-Nexus Enterprise Deployment Guide
## Production-Ready Deployment with Security, Monitoring & Auto-scaling

**Document Version**: 1.0  
**Last Updated**: December 3, 2025  
**Target Audience**: DevOps Engineers, Platform Engineers, Security Teams

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Security Configuration](#security-configuration)
4. [Monitoring Setup](#monitoring-setup)
5. [Cloud Deployment](#cloud-deployment)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Cost Optimization](#cost-optimization)

---

## Overview

This guide covers enterprise-grade deployment of AI-Nexus with:

- **Security**: JWT auth, RBAC, rate limiting, input validation
- **Monitoring**: Prometheus metrics (P50/P95/P99), cost tracking, alerts
- **Optimization**: Model caching, batch processing, auto-scaling
- **Scale Testing**: Load testing suite (1K+ req/s capability)
- **Cloud Deployment**: AWS EKS & GCP GKE configurations

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer (NLB/GCE)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  API Pod 1   │ │  API Pod 2  │ │  API Pod N  │
│ (Auth+RL)    │ │ (Auth+RL)   │ │ (Auth+RL)   │
└──────┬───────┘ └──────┬──────┘ └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
│  LLM Pod 1   │ │ Vector DB │ │ Redis Cache │
│ (GPU: T4/A100│ │ (Qdrant)  │ │ (Sessions)  │
└──────────────┘ └───────────┘ └─────────────┘
        │               │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │  Prometheus   │
        │  (Metrics)    │
        └───────────────┘
```

---

## Prerequisites

### Required Tools
```bash
# Kubernetes
kubectl version  # >= 1.28
helm version     # >= 3.0

# Cloud CLI
aws --version    # For AWS deployment
gcloud version   # For GCP deployment

# Python
python --version # >= 3.10
pip --version

# Docker
docker --version # >= 24.0
```

### Hardware Requirements

| Component | Minimum | Recommended | Enterprise |
|-----------|---------|-------------|------------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 16 GB | 32 GB | 64+ GB |
| **GPU** | T4 (16GB) | V100 (32GB) | A100 (40GB) |
| **Storage** | 100 GB SSD | 500 GB SSD | 1+ TB NVMe |
| **Network** | 1 Gbps | 10 Gbps | 25+ Gbps |

---

## Security Configuration

### 1. JWT Authentication

**Generate Secure Keys:**
```bash
# Generate JWT secret (store in environment variable or secrets manager)
openssl rand -base64 64 > jwt_secret.key

# Set in environment
export JWT_SECRET=$(cat jwt_secret.key)
```

**Create Admin User:**
```python
from core.security import get_password_manager, User, Role

password_manager = get_password_manager()

# Hash password
hashed_password = password_manager.hash_password("SecureAdminPass123!")

# Create admin user
admin_user = UserInDB(
    username="admin",
    email="admin@ainexus.com",
    hashed_password=hashed_password,
    role=Role.ADMIN
)

# Store in database (implement your user store)
```

**API Key Generation:**
```python
from core.security import get_api_key_manager, Role

api_key_manager = get_api_key_manager()

# Generate API key for programmatic access
raw_key, api_key = api_key_manager.generate_api_key(
    user_id="service_account_1",
    name="Production API Key",
    role=Role.DEVELOPER,
    expires_days=90
)

print(f"API Key (save this!): {raw_key}")
# Output: nexus_Ab3Dk9X2_vF4hK...
```

### 2. Rate Limiting

**Configure Redis:**
```bash
# Deploy Redis for distributed rate limiting
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis \
  --set auth.password="your-secure-password" \
  --namespace ai-nexus
```

**Set Rate Limits:**
```python
from core.rate_limiting import RateLimitConfig

config = RateLimitConfig(
    redis_host="redis-master.ai-nexus.svc.cluster.local",
    redis_password="your-secure-password",
    # Free tier
    requests_per_minute=10,
    requests_per_hour=100,
    # Enterprise tier (configure in RATE_LIMIT_TIERS)
)
```

### 3. Input Validation

**Enable Strict Validation:**
```python
from core.validation import ValidatedModelInput

# All API endpoints should use validated models
@app.post("/api/llm/generate")
async def generate(input: ValidatedModelInput):
    # Input is automatically sanitized
    # XSS, SQL injection, path traversal blocked
    result = await llm_service.generate(input.prompt)
    return result
```

### 4. Kubernetes Secrets

```bash
# Create secrets for sensitive data
kubectl create secret generic ai-nexus-secrets \
  --from-literal=jwt-secret="$(cat jwt_secret.key)" \
  --from-literal=redis-password="your-redis-password" \
  --from-literal=qdrant-api-key="your-qdrant-key" \
  --namespace ai-nexus

# Verify
kubectl get secret ai-nexus-secrets -o yaml -n ai-nexus
```

---

## Monitoring Setup

### 1. Prometheus & Grafana

**Install Prometheus Stack:**
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword="admin"
```

**Access Grafana:**
```bash
# Port forward
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Login: http://localhost:3000
# Username: admin
# Password: admin (change immediately!)
```

**Import AI-Nexus Dashboard:**
1. Navigate to Dashboards → Import
2. Upload `docs/grafana-dashboard.json`
3. Select Prometheus data source

**Key Metrics:**
```yaml
# CPU & Memory
- ainexus_cpu_usage_percent
- ainexus_memory_usage_bytes

# LLM Inference
- ainexus_llm_inference_duration_seconds (P50, P95, P99)
- ainexus_llm_tokens_per_second
- ainexus_llm_input_tokens_total
- ainexus_llm_output_tokens_total

# Costs
- ainexus_cost_total_dollars

# Errors
- ainexus_errors_total
- ainexus_http_requests_total{status="500"}
```

### 2. Cost Tracking

**Configure Budget Alerts:**
```python
from core.usage_tracking import get_token_tracker, BudgetConfig

tracker = get_token_tracker()

# Set budgets
tracker.budget = BudgetConfig(
    daily_limit=100.0,    # $100/day
    monthly_limit=2000.0, # $2000/month
    alert_threshold=0.8   # Alert at 80%
)
```

**View Cost Dashboard:**
```bash
# Get cost by user
python -c "
from core.usage_tracking import get_token_tracker
tracker = get_token_tracker()
print(tracker.get_top_users(limit=10))
"

# Get cost by model
python -c "
from core.usage_tracking import get_token_tracker
tracker = get_token_tracker()
print(tracker.get_top_models(limit=10))
"
```

### 3. Alerting

**Configure Alert Manager:**
```python
from core.alerting import AlertConfig

config = AlertConfig(
    error_threshold=10,        # Alert after 10 errors
    critical_threshold=5,      # Alert after 5 critical errors
    time_window_minutes=5,     # In 5-minute window
    max_alerts_per_hour=10,    # Rate limit alerts
    enable_console=True,       # Console alerts
    # TODO: Configure external channels
    # enable_slack=True,
    # enable_pagerduty=True,
)
```

**Custom Alert Handler:**
```python
from core.alerting import get_alert_manager, Alert

def slack_alert_handler(alert: Alert):
    # Send to Slack webhook
    import requests
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    
    requests.post(webhook_url, json={
        "text": f"🚨 {alert.title}",
        "attachments": [{
            "color": "danger" if alert.severity == "critical" else "warning",
            "text": alert.message,
            "fields": [
                {"title": "Count", "value": str(alert.count), "short": True},
                {"title": "Severity", "value": alert.severity.upper(), "short": True}
            ]
        }]
    })

# Register handler
alert_manager = get_alert_manager()
alert_manager.add_handler(slack_alert_handler)
```

---

## Cloud Deployment

### AWS EKS Deployment

**Full instructions**: See `deploy/aws-deployment.sh`

**Quick Start:**
```bash
# 1. Create EKS cluster with GPU
eksctl create cluster \
  --name ai-nexus-cluster \
  --region us-east-1 \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5

# 2. Install NVIDIA plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml

# 3. Deploy AI-Nexus
kubectl create namespace ai-nexus
kubectl apply -f k8s/ -n ai-nexus

# 4. Get Load Balancer URL
kubectl get svc ai-nexus-lb -n ai-nexus
```

**Cost Estimate (p3.2xlarge):**
- On-Demand: $3.06/hour → ~$2,200/month
- Spot: ~$0.90/hour → ~$650/month (70% savings!)

### GCP GKE Deployment

**Full instructions**: See `deploy/gcp-deployment.sh`

**Quick Start:**
```bash
# 1. Create GKE cluster with GPU
gcloud container clusters create ai-nexus-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=5

# 2. Install NVIDIA driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# 3. Deploy AI-Nexus
kubectl create namespace ai-nexus
kubectl apply -f k8s/ -n ai-nexus

# 4. Get Load Balancer IP
kubectl get svc ai-nexus-lb -n ai-nexus
```

**Cost Estimate (n1-standard-4 + T4):**
- On-Demand: ~$0.95/hour → ~$690/month
- Spot: ~$0.25/hour → ~$180/month (74% savings!)

---

## Performance Optimization

### 1. Model Caching

**Enable Multi-Level Cache:**
```python
from services.optimization.caching import CacheConfig, get_model_cache

# Configure cache
config = CacheConfig(
    max_memory_items=1000,
    max_memory_size_mb=500,
    enable_disk_cache=True,
    disk_cache_dir="/data/cache",
    max_disk_size_mb=5000,
    default_ttl_seconds=3600
)

cache = get_model_cache()

# Cache embeddings
@cached(ttl=3600, key_func=lambda text: f"embed:{hash(text)}")
def embed_text(text):
    return embedding_service.embed(text)
```

**Cache Performance:**
- L1 (Memory): <1ms latency
- L2 (Disk): ~10ms latency
- Cache Hit Rate: 60-80% typical

### 2. Batch Processing

**Configure Dynamic Batching:**
```python
from services.optimization.batching import BatchConfig, get_batch_manager

config = BatchConfig(
    max_batch_size=32,
    min_batch_size=1,
    max_wait_ms=100,
    enable_auto_tuning=True,
    target_latency_ms=1000
)

# Batch manager automatically groups requests
manager = get_batch_manager()
```

**Batch Performance:**
- Throughput: 5-10x improvement
- Latency: <200ms added overhead
- GPU Utilization: 85-95% (vs 30-40% without batching)

### 3. Auto-Scaling

**Horizontal Pod Autoscaler:**
```bash
# Apply HPA configs
kubectl apply -f k8s/hpa.yaml -n ai-nexus

# View HPA status
kubectl get hpa -n ai-nexus

# Example output:
# NAME                  REFERENCE                  TARGETS   MIN   MAX   REPLICAS
# ai-nexus-llm-hpa      Deployment/ai-nexus-llm    45%/70%   2     10    4
```

**Cluster Autoscaler:**
- Automatically adds/removes nodes
- Based on pod resource requests
- Configured in AWS/GCP deployment scripts

---

## Scale Testing

### Load Testing with Locust

**Run Load Tests:**
```bash
# Light load (10 users)
python tests/load_tests.py --host http://localhost:8000 --scenario light

# Moderate load (50 users)
python tests/load_tests.py --host http://localhost:8000 --scenario moderate

# Stress test (200 users)
python tests/load_tests.py --host http://localhost:8000 --scenario stress

# With Web UI
locust -f tests/load_tests.py --host http://localhost:8000
# Open: http://localhost:8089
```

**Performance Targets:**
| Metric | Target | Excellent |
|--------|--------|-----------|
| **Throughput** | 100 req/s | 500+ req/s |
| **P50 Latency** | <500ms | <200ms |
| **P95 Latency** | <2s | <1s |
| **P99 Latency** | <5s | <2s |
| **Error Rate** | <1% | <0.1% |
| **GPU Utilization** | >70% | >85% |

### Memory Leak Detection

```bash
# Run with memory profiler
pytest tests/test_enterprise.py::TestPerformance \
  --memray \
  --duration=3600  # 1 hour test

# Check for memory growth
python -m memory_profiler services/ml/llm_service.py
```

---

## Troubleshooting

### Common Issues

**1. GPU Not Detected**
```bash
# Check GPU availability
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# If 0 GPUs, reinstall driver
kubectl delete daemonset nvidia-device-plugin-daemonset -n kube-system
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
```

**2. High Latency**
```bash
# Check metrics
kubectl top pods -n ai-nexus

# View logs
kubectl logs -f deployment/ai-nexus-llm -n ai-nexus

# Check batch processing stats
python -c "
from services.optimization.batching import get_batch_manager
print(get_batch_manager().get_global_stats())
"
```

**3. Rate Limit Issues**
```python
from core.rate_limiting import get_rate_limiter
import asyncio

async def reset_user_limit():
    limiter = await get_rate_limiter()
    await limiter.reset_limit("user123")

asyncio.run(reset_user_limit())
```

### Logs & Debugging

```bash
# View all logs
kubectl logs -l app=ai-nexus -n ai-nexus --tail=100

# Stream logs
stern -n ai-nexus ai-nexus

# Check errors
kubectl logs -n ai-nexus deployment/ai-nexus-api | grep ERROR

# Exec into pod
kubectl exec -it deployment/ai-nexus-llm -n ai-nexus -- bash
```

---

## Cost Optimization

### GPU Spot Instances

**AWS:**
- Savings: 60-90% off on-demand
- Interruption rate: ~5% (varies by region)
- Best for: Stateless inference workloads

**GCP:**
- Spot VMs: Up to 91% off
- Preemptible VMs: Up to 80% off  
- Best practice: Mix spot + on-demand for reliability

### Auto-Scaling Best Practices

```yaml
# Aggressive scale-down for cost savings
scaleDown:
  stabilizationWindowSeconds: 180  # Wait 3min before scaling down
  policies:
  - type: Percent
    value: 50  # Scale down 50% at a time
    periodSeconds: 60

# Fast scale-up for performance
scaleUp:
  stabilizationWindowSeconds: 30  # React quickly
  policies:
  - type: Pods
    value: 5  # Add 5 pods at a time
    periodSeconds: 15
```

### Budget Monitoring

**Set Up Cost Alerts:**
```bash
# AWS CloudWatch Budget
aws budgets create-budget \
  --account-id <ACCOUNT_ID> \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json

# GCP Budget Alert
gcloud billing budgets create \
  --billing-account=<BILLING_ACCOUNT_ID> \
  --display-name="AI-Nexus Monthly Budget" \
  --budget-amount=1000USD \
  --threshold-rule=percent=80
```

---

## Appendix

### A. Test Checklist

Before production deployment:

- [ ] All 190+ tests passing
- [ ] Load test completed (1K+ req/s)
- [ ] Memory leak test passed (1+ hour)
- [ ] Security audit completed
- [ ] Disaster recovery tested
- [ ] Monitoring dashboards configured
- [ ] Alert handlers tested
- [ ] Documentation updated
- [ ] Backup strategy in place
- [ ] Cost alerts configured

### B. Security Checklist

- [ ] JWT secret rotated
- [ ] TLS/SSL enabled
- [ ] Rate limiting configured
- [ ] Input validation enabled
- [ ] RBAC roles defined
- [ ] API keys expire after 90 days
- [ ] Secrets stored in vault/secrets manager
- [ ] Network policies applied
- [ ] Pod security policies enforced
- [ ] Vulnerability scanning enabled

### C. Support

- **Documentation**: `docs/` directory
- **Issues**: GitHub Issues
- **Weekly Reports**: `docs/WEEK*_REPORT.md`
- **Architecture**: `PROJECT_COMPLETE.md`

---

**🎉 Congratulations!** You now have an enterprise-grade AI platform ready for production deployment.

For questions or issues, refer to the comprehensive test suite in `tests/test_enterprise.py` and load testing guide in `tests/load_tests.py`.

**Happy Deploying! 🚀**
