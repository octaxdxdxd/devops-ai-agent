from __future__ import annotations

from src.tools.output import compress_output


def test_compress_output_summarizes_large_deployment_manifest_preserving_key_spec_fields() -> None:
    manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: demo-api
  namespace: apps
  labels:
    app: demo-api
  annotations:
    long.example.com/blob: "%s"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: demo-api
  template:
    metadata:
      labels:
        app: demo-api
    spec:
      serviceAccountName: demo-api
      nodeSelector:
        role: application
      containers:
      - name: web
        image: ghcr.io/example/demo-api:1.2.3
        ports:
        - name: http
          containerPort: 8080
        env:
        - name: APP_ENV
          value: production
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: password
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /live
            port: 8080
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: cache
          mountPath: /cache
      volumes:
      - name: config
        configMap:
          name: demo-api-config
      - name: cache
        emptyDir: {}
""" % ("x" * 9000)

    result = compress_output(manifest, max_lines=40, max_chars=1200, format_hint="k8s_manifest")

    assert "kind: Deployment" in result
    assert "annotation_keys:" in result
    assert "spec_summary:" in result
    assert "resources:" in result
    assert "requests:" in result
    assert 'cpu: 500m' in result
    assert 'memory: 512Mi' in result
    assert "readinessProbe:" in result
    assert "livenessProbe:" in result
    assert "volumeMounts:" in result
    assert "valueFrom:" in result
    assert "long.example.com/blob" in result
    assert "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" not in result


def test_compress_output_auto_detects_and_summarizes_non_workload_manifest() -> None:
    manifest = """
apiVersion: v1
kind: Service
metadata:
  name: demo-api
  namespace: apps
spec:
  type: ClusterIP
  selector:
    app: demo-api
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
""" + ("\n# filler\n" * 300)

    result = compress_output(manifest, max_lines=25, max_chars=500)

    assert "kind: Service" in result
    assert "spec_summary:" in result
    assert "selector:" in result
    assert "ports:" in result
    assert "targetPort: 8080" in result


def test_compress_output_preserves_named_sections_for_exec_output() -> None:
    output = "\n".join(
        [
            "/opt/sonatype",
            "--- / ---",
            "root-file-1",
            "root-file-2",
            "root-file-3",
            "root-file-4",
            "--- /nexus-data ---",
            "db",
            "blobs",
            "tmp",
            "restore-from-backup",
            "--- /opt ---",
            "sonatype",
            "plugins",
            "java",
            "--- /tmp ---",
            "hsperfdata_nexus",
            "cache",
            "pid",
        ]
    )

    result = compress_output(output, max_lines=12, max_chars=260, format_hint="sectioned_text")

    assert "--- / ---" in result
    assert "--- /nexus-data ---" in result
    assert "--- /opt ---" in result
    assert "--- /tmp ---" in result
    assert "root-file-1" in result
    assert "db" in result
    assert "sonatype" in result
    assert "hsperfdata_nexus" in result
