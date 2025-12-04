# ============================================================================
# AWS Deployment Configuration
# ============================================================================

# Prerequisites:
# - AWS CLI configured
# - kubectl configured
# - eksctl installed
# - GPU-enabled EC2 instances (p3.2xlarge or p3.8xlarge recommended)

# 1. Create EKS Cluster with GPU nodes
# eksctl create cluster \
#   --name ai-nexus-cluster \
#   --region us-east-1 \
#   --node-type p3.2xlarge \
#   --nodes 2 \
#   --nodes-min 1 \
#   --nodes-max 5 \
#   --with-oidc \
#   --ssh-access \
#   --ssh-public-key your-key-name \
#   --managed

# 2. Install NVIDIA Device Plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml

# 3. Install Metrics Server (for HPA)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 4. Install Prometheus (for custom metrics)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# 5. Create namespace
kubectl create namespace ai-nexus

# 6. Create secrets
kubectl create secret generic ai-nexus-secrets \
  --from-literal=jwt-secret="your-jwt-secret-key" \
  --from-literal=redis-password="your-redis-password" \
  --namespace ai-nexus

# 7. Deploy AI-Nexus
kubectl apply -f k8s/ --namespace ai-nexus

# 8. Create EBS storage class for persistent volumes
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ai-nexus-storage
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  fsType: ext4
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF

# 9. Create Load Balancer
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ai-nexus-lb
  namespace: ai-nexus
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  selector:
    app: ai-nexus-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
EOF

# 10. Get Load Balancer URL
kubectl get service ai-nexus-lb -n ai-nexus -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# ============================================================================
# AWS Cost Optimization
# ============================================================================

# Use Spot Instances (60-90% cheaper)
cat <<EOF > spot-nodegroup.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ai-nexus-cluster
  region: us-east-1

managedNodeGroups:
  - name: gpu-spot-nodes
    instanceTypes:
      - p3.2xlarge
      - p3.8xlarge
    spot: true
    minSize: 1
    maxSize: 5
    desiredCapacity: 2
    volumeSize: 100
    volumeType: gp3
    ssh:
      allow: true
      publicKeyName: your-key-name
    labels:
      workload: gpu
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
EOF

# Apply spot instances
eksctl create nodegroup -f spot-nodegroup.yaml

# ============================================================================
# Monitoring Setup
# ============================================================================

# Install Grafana dashboards
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Default credentials:
# Username: admin
# Password: prom-operator

# Import AI-Nexus dashboard (dashboard ID: 12345)

# ============================================================================
# Auto-scaling Configuration
# ============================================================================

# Cluster Autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Configure for your cluster
kubectl -n kube-system edit deployment cluster-autoscaler

# Add flags:
# --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/ai-nexus-cluster
# --balance-similar-node-groups
# --skip-nodes-with-system-pods=false

# ============================================================================
# Backup Configuration
# ============================================================================

# Install Velero for backups
helm repo add vmware-tanzu https://vmware-tanzu.github.io/helm-charts
helm install velero vmware-tanzu/velero \
  --namespace velero \
  --create-namespace \
  --set configuration.provider=aws \
  --set configuration.backupStorageLocation.bucket=ai-nexus-backups \
  --set configuration.backupStorageLocation.config.region=us-east-1 \
  --set credentials.useSecret=true \
  --set credentials.secretContents.cloud=<AWS_CREDENTIALS>

# Schedule daily backups
velero schedule create daily-backup --schedule="0 2 * * *"

# ============================================================================
# SSL/TLS Configuration
# ============================================================================

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# ============================================================================
# Performance Tuning
# ============================================================================

# Enable NVIDIA MIG (Multi-Instance GPU) for better utilization
# For A100 GPUs:
sudo nvidia-smi -mig 1
sudo nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C

# Configure kubelet for GPU
cat <<EOF > /etc/systemd/system/kubelet.service.d/11-nvidia.conf
[Service]
Environment="KUBELET_EXTRA_ARGS=--feature-gates=DevicePlugins=true"
EOF

systemctl daemon-reload
systemctl restart kubelet

# ============================================================================
# Troubleshooting
# ============================================================================

# Check GPU availability
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# View pod logs
kubectl logs -f deployment/ai-nexus-llm -n ai-nexus

# Describe pod for issues
kubectl describe pod <pod-name> -n ai-nexus

# Check HPA status
kubectl get hpa -n ai-nexus

# View metrics
kubectl top nodes
kubectl top pods -n ai-nexus
