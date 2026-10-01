FROM registry.redhat.io/rhoai/odh-workbench-jupyter-datascience-cpu-py311-rhel9:v2.24.0-1757341840

USER 0
COPY wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels \
    "mlflow[kubernetes]>=3.11" "packaging>=23"
USER 1001
