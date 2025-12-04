# Kubernetes Deployment Guide - AI-Nexus GPU Platform

## Overview

This guide covers deploying AI-Nexus on Kubernetes with GPU support, distributed training, and comprehensive monitoring.

## Prerequisites

### Required Software
- **Kubernetes cluster** (v1.24+)
- **Helm** (v3.0+)
- **kubectl** configured
- **NVIDIA GPU nodes** with:
  - NVIDIA drivers installed
  - Container runtime with GPU support (e.g., containerd with nvidia-container-runtime)

### Cluster Requirements
- Minimum 3 nodes (1 master, 2 workers)
- At least 1 GPU per worker node
- 100GB+ storage per node
- Network policy support (Calico/Weave/Cilium)

## Quick Start

### 1. Install NVIDIA Device Plugin

Deploy the NVIDIA device plugin to enable GPU scheduling:

```bash
kubectl apply -f k8s/manifests/nvidia-device-plugin.yaml
```

Verify GPUs are detected:

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable | select(.["nvidia.com/gpu"] != null)'
```

### 2. Create Namespace and RBAC

```bash
kubectl apply -f k8s/manifests/namespace.yaml
kubectl apply -f k8s/manifests/rbac.yaml
kubectl apply -f k8s/manifests/priority-class.yaml
kubectl apply -f k8s/manifests/resource-quota.yaml
kubectl apply -f k8s/manifests/network-policies.yaml
```

### 3. Deploy Monitoring Stack

Deploy Prometheus and Grafana for GPU metrics:

```bash
kubectl apply -f k8s/manifests/monitoring/prometheus.yaml
kubectl apply -f k8s/manifests/monitoring/dcgm-exporter.yaml
kubectl apply -f k8s/manifests/monitoring/grafana.yaml
```

Access Grafana:

```bash
kubectl port-forward -n ai-nexus svc/grafana 3000:3000
```

Open http://localhost:3000 (admin/change-this-password)

### 4. Deploy AI-Nexus with Helm

```bash
# Update Helm dependencies
helm dependency update k8s/helm/ai-nexus

# Install the chart
helm install ai-nexus k8s/helm/ai-nexus \
  --namespace ai-nexus \
  --create-namespace \
  --values k8s/helm/ai-nexus/values.yaml

# Check deployment status
kubectl get pods -n ai-nexus -w
```

## Configuration

### GPU Settings

Edit `k8s/helm/ai-nexus/values.yaml`:

```yaml
gpu:
  enabled: true
  count: 2  # GPUs per pod
  memory: "16Gi"  # GPU memory requirement
  vendor: nvidia
```

### Distributed Training

Configure multi-node training:

```yaml
distributed:
  enabled: true
  backend: nccl
  worldSize: 4
  masterPort: 29500
```

### Autoscaling

Configure GPU-based autoscaling:

```yaml
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: nvidia.com/gpu
        target:
          type: Utilization
          averageUtilization: 80
```

## Deployment Verification

### Check Pod Status

```bash
kubectl get pods -n ai-nexus
kubectl describe pod <pod-name> -n ai-nexus
```

### Verify GPU Access

```bash
kubectl exec -it <pod-name> -n ai-nexus -- python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

### Check Logs

```bash
kubectl logs -f <pod-name> -n ai-nexus
```

### Monitor GPU Metrics

```bash
# Port forward Prometheus
kubectl port-forward -n ai-nexus svc/prometheus 9090:9090

# Port forward DCGM Exporter
kubectl port-forward -n ai-nexus svc/dcgm-exporter 9400:9400
```

Query GPU metrics:
- http://localhost:9090 (Prometheus)
- http://localhost:9400/metrics (DCGM raw metrics)

## Distributed Training

### Multi-Node Training Job

Create a training job:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: distributed-training
  namespace: ai-nexus
spec:
  parallelism: 4
  completions: 4
  template:
    metadata:
      labels:
        app.kubernetes.io/component: training
    spec:
      serviceAccountName: ai-nexus-sa
      containers:
      - name: trainer
        image: ai-nexus:latest
        command: ["python", "-m", "services.ml.distributed_train"]
        env:
        - name: WORLD_SIZE
          value: "4"
        - name: MASTER_ADDR
          value: "ai-nexus-distributed-master"
        - name: MASTER_PORT
          value: "29500"
        - name: NCCL_DEBUG
          value: "INFO"
        resources:
          limits:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: "32Gi"
```

Apply:

```bash
kubectl apply -f distributed-training-job.yaml
kubectl logs -f job/distributed-training -n ai-nexus
```

## Monitoring and Observability

### Prometheus Queries

GPU utilization:

```promql
avg(DCGM_FI_DEV_GPU_UTIL) by (gpu, UUID)
```

GPU memory usage:

```promql
DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE * 100
```

GPU temperature:

```promql
DCGM_FI_DEV_GPU_TEMP
```

### Grafana Dashboards

Import pre-built dashboards:

1. NVIDIA DCGM Exporter Dashboard (ID: 12239)
2. Kubernetes GPU Metrics (custom)

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA device plugin
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds

# Verify node has GPUs
kubectl describe node <node-name> | grep nvidia.com/gpu

# Check driver version
kubectl exec -it <pod-name> -n ai-nexus -- nvidia-smi
```

### Out of Memory Errors

Reduce batch size or GPU count:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1  # Reduce from 2 to 1
```

### Network Issues in Distributed Training

Check network policies:

```bash
kubectl get networkpolicy -n ai-nexus
kubectl describe networkpolicy ai-nexus-distributed-training -n ai-nexus
```

Verify NCCL communication:

```bash
kubectl exec -it <pod-name> -n ai-nexus -- \
  python -c "import torch.distributed as dist; dist.init_process_group(backend='nccl')"
```

### Pod Stuck in Pending

Check resource availability:

```bash
kubectl describe pod <pod-name> -n ai-nexus | grep -A 10 Events
```

Common causes:
- Insufficient GPU resources
- Resource quota exceeded
- Node selector not matching
- PVC not bound

## Scaling

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment ai-nexus -n ai-nexus --replicas=5

# Scale distributed training
kubectl scale job distributed-training -n ai-nexus --replicas=8
```

### Auto-scaling

Horizontal Pod Autoscaler is configured automatically:

```bash
kubectl get hpa -n ai-nexus
kubectl describe hpa ai-nexus -n ai-nexus
```

## Backup and Recovery

### Backup Models

Models are stored in persistent volumes:

```bash
kubectl get pvc -n ai-nexus
kubectl exec -it <pod-name> -n ai-nexus -- tar -czf /tmp/models.tar.gz /app/models
kubectl cp ai-nexus/<pod-name>:/tmp/models.tar.gz ./models-backup.tar.gz
```

### Restore from Backup

```bash
kubectl cp ./models-backup.tar.gz ai-nexus/<pod-name>:/tmp/models.tar.gz
kubectl exec -it <pod-name> -n ai-nexus -- tar -xzf /tmp/models.tar.gz -C /app
```

## Upgrading

### Helm Upgrade

```bash
# Update values
vim k8s/helm/ai-nexus/values.yaml

# Upgrade release
helm upgrade ai-nexus k8s/helm/ai-nexus \
  --namespace ai-nexus \
  --values k8s/helm/ai-nexus/values.yaml

# Rollback if needed
helm rollback ai-nexus -n ai-nexus
```

### Rolling Update

```bash
# Update image
kubectl set image deployment/ai-nexus ai-nexus=ai-nexus:v2.0 -n ai-nexus

# Check rollout status
kubectl rollout status deployment/ai-nexus -n ai-nexus

# Rollback
kubectl rollout undo deployment/ai-nexus -n ai-nexus
```

## Production Best Practices

1. **Resource Limits**: Always set resource limits to prevent resource exhaustion
2. **GPU Sharing**: Use MIG (Multi-Instance GPU) for better utilization
3. **Monitoring**: Set up alerts for GPU utilization, temperature, and errors
4. **Backup**: Regular backups of models and data
5. **Security**: Use network policies, RBAC, and pod security policies
6. **Cost Optimization**: Use spot instances for training jobs
7. **Logging**: Centralized logging with ELK or Loki
8. **CI/CD**: Automated testing and deployment

## Performance Tuning

### NCCL Optimization

```yaml
env:
- name: NCCL_IB_DISABLE
  value: "0"  # Enable InfiniBand
- name: NCCL_DEBUG
  value: "WARN"  # Reduce logging overhead
- name: NCCL_SOCKET_IFNAME
  value: "eth0"  # Specify network interface
```

### GPU Memory Optimization

```yaml
env:
- name: PYTORCH_CUDA_ALLOC_CONF
  value: "max_split_size_mb:128"
```

### Batch Size Tuning

Use the GPU scheduler to calculate optimal batch size:

```python
from services.gpu.scheduler import get_scheduler

scheduler = get_scheduler()
batch_size = scheduler.get_optimal_batch_size(model, input_shape, gpu_id=0)
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/thequantumfalcon/AI-Nexus/issues
- Documentation: See WEEK2_KUBERNETES_PLAN.md
- Logs: `kubectl logs -n ai-nexus <pod-name>`

## Clean Up

Remove all resources:

```bash
helm uninstall ai-nexus -n ai-nexus
kubectl delete namespace ai-nexus
kubectl delete -f k8s/manifests/nvidia-device-plugin.yaml
```
