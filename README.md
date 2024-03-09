# deep Classifier project
LINK TO APP: https://classifywithcnns.streamlit.app/
or https://classifywithcnn.onrender.com/
## workflow

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config.
6. Update the components
7. Update the pipeline
8. Test run pipeline stage
9. run tox for testing your package
10. Update the dvc.yaml
11. run "dvc repro" for running all the stages in pipeline

![img](https://dagshub.com/PARADOXop/DEEPCNNClassifier/src/master/docs/images/Data%20Ingestion@2x%20%281%29.png)

### MLflow credentials are added into vars.py file to keep them safe  

STEP 1: Set the env variable | Get it from dagshub -> remote tab -> mlflow tab

STEP 2: install mlflow

STEP 3: Set remote URI

STEP 4: Use context manager of mlflow to start run and then log metrics, params and model


mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0 -p 1234

conda activate C:/Users/rkuka/miniconda3
conda activate P:/DEEPCNNClassifier/env
