# ============================================================================
# GCP Deployment Configuration
# ============================================================================

# Prerequisites:
# - gcloud CLI configured
# - kubectl installed
# - GKE cluster with GPU support

# 1. Create GKE Cluster with GPU nodes
gcloud container clusters create ai-nexus-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=5 \
  --enable-autorepair \
  --enable-autoupgrade \
  --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver

# 2. Get cluster credentials
gcloud container clusters get-credentials ai-nexus-cluster --zone=us-central1-a

# 3. Install NVIDIA GPU driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# 4. Install Metrics Server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 5. Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# 6. Create namespace
kubectl create namespace ai-nexus

# 7. Create secrets
kubectl create secret generic ai-nexus-secrets \
  --from-literal=jwt-secret="$(openssl rand -base64 32)" \
  --from-literal=redis-password="$(openssl rand -base64 32)" \
  --namespace ai-nexus

# 8. Deploy AI-Nexus
kubectl apply -f k8s/ --namespace ai-nexus

# 9. Create persistent storage
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ai-nexus-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: regional-pd
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF

# 10. Create Load Balancer with static IP
# Reserve static IP
gcloud compute addresses create ai-nexus-ip --global

# Get the IP
gcloud compute addresses describe ai-nexus-ip --global --format="get(address)"

# Create service with static IP
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ai-nexus-lb
  namespace: ai-nexus
  annotations:
    cloud.google.com/load-balancer-type: "External"
spec:
  type: LoadBalancer
  loadBalancerIP: <STATIC_IP_FROM_ABOVE>
  selector:
    app: ai-nexus-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
EOF

# ============================================================================
# GCP Cost Optimization
# ============================================================================

# Use Preemptible VMs (up to 80% cheaper)
gcloud container node-pools create gpu-preemptible-pool \
  --cluster=ai-nexus-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=5 \
  --preemptible \
  --node-labels=workload=gpu \
  --node-taints=nvidia.com/gpu=true:NoSchedule

# Enable Spot VMs (even cheaper)
gcloud container node-pools create gpu-spot-pool \
  --cluster=ai-nexus-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=5 \
  --spot \
  --node-labels=workload=gpu

# ============================================================================
# GPU Options and Pricing (as of 2025)
# ============================================================================

# T4 (16GB) - Best for inference
# Machine: n1-standard-4 with 1x T4
# ~$0.35/hour on-demand, ~$0.07/hour spot
# Good for: Production inference, 100+ req/s

# V100 (16GB) - Balanced
# Machine: n1-standard-8 with 1x V100
# ~$2.48/hour on-demand, ~$0.74/hour spot
# Good for: Mixed workloads, fine-tuning

# A100 (40GB) - Best for training
# Machine: a2-highgpu-1g with 1x A100
# ~$3.67/hour on-demand, ~$1.10/hour spot
# Good for: Large model training, 70B+ LLMs

# ============================================================================
# Monitoring Setup
# ============================================================================

# Install Cloud Monitoring
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com

# Create uptime check
gcloud monitoring uptime-checks create http ai-nexus-uptime \
  --resource-type=uptime-url \
  --host=<YOUR_DOMAIN> \
  --path=/health

# Create alerting policy
gcloud alpha monitoring policies create \
  --notification-channels=<CHANNEL_ID> \
  --display-name="AI-Nexus High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=5 \
  --condition-threshold-duration=60s

# ============================================================================
# Auto-scaling Configuration
# ============================================================================

# Enable cluster autoscaler (already enabled in create command)
# Configure node pool autoscaling
gcloud container clusters update ai-nexus-cluster \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --zone=us-central1-a \
  --node-pool=default-pool

# Vertical Pod Autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/vertical-pod-autoscaler/deploy/vpa-v1-crd-gen.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/vertical-pod-autoscaler/deploy/vpa-rbac.yaml

# ============================================================================
# Backup Configuration
# ============================================================================

# Enable GKE backup
gcloud container clusters update ai-nexus-cluster \
  --enable-backup-and-restore \
  --zone=us-central1-a

# Create backup plan
cat <<EOF | kubectl apply -f -
apiVersion: gkebackup.gke.io/v1
kind: BackupPlan
metadata:
  name: ai-nexus-backup-plan
  namespace: ai-nexus
spec:
  cluster: ai-nexus-cluster
  backupSchedule:
    cronSchedule: "0 2 * * *"
  retentionPolicy:
    backupRetainDays: 30
    locked: false
EOF

# ============================================================================
# SSL/TLS Configuration
# ============================================================================

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create managed certificate (GCP-specific)
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: ai-nexus-cert
  namespace: ai-nexus
spec:
  domains:
    - api.ainexus.com
    - www.ainexus.com
EOF

# Update ingress to use certificate
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-nexus-ingress
  namespace: ai-nexus
  annotations:
    kubernetes.io/ingress.class: "gce"
    networking.gke.io/managed-certificates: ai-nexus-cert
    kubernetes.io/ingress.global-static-ip-name: ai-nexus-ip
spec:
  rules:
  - host: api.ainexus.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: ai-nexus-api
            port:
              number: 8000
EOF

# ============================================================================
# Performance Tuning
# ============================================================================

# Enable GKE image streaming (faster cold starts)
gcloud container clusters update ai-nexus-cluster \
  --enable-image-streaming \
  --zone=us-central1-a

# Enable workload identity (better security)
gcloud container clusters update ai-nexus-cluster \
  --workload-pool=PROJECT_ID.svc.id.goog \
  --zone=us-central1-a

# ============================================================================
# Cost Monitoring
# ============================================================================

# Enable detailed billing export
gcloud services enable billingbudgets.googleapis.com

# Create budget alert
gcloud billing budgets create \
  --billing-account=<BILLING_ACCOUNT_ID> \
  --display-name="AI-Nexus Monthly Budget" \
  --budget-amount=1000USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100

# ============================================================================
# Troubleshooting
# ============================================================================

# Check GPU availability
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# View GKE cluster details
gcloud container clusters describe ai-nexus-cluster --zone=us-central1-a

# Check logs in Cloud Logging
gcloud logging read "resource.type=k8s_container AND resource.labels.namespace_name=ai-nexus" --limit=50

# View metrics in Cloud Monitoring
gcloud monitoring dashboards list

# SSH into node for debugging
gcloud compute ssh <NODE_NAME> --zone=us-central1-a

# ============================================================================
# Cleanup
# ============================================================================

# Delete cluster (WARNING: This deletes everything)
gcloud container clusters delete ai-nexus-cluster --zone=us-central1-a

# Release static IP
gcloud compute addresses delete ai-nexus-ip --global
