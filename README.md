# deep Classifier project

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


STEP 1: Set the env variable | Get it from dagshub -> remote tab -> mlflow tab

STEP 2: install mlflow

STEP 3: Set remote URI

STEP 4: Use context manager of mlflow to start run and then log metrics, params and model



1. Real-time video processing using opencv
2. We load model while processing the video to detect vehicles
3. Distance Measurements: Using Monocular depth estimation to extimate distance accurately because we using only one cam otherwise 
4. Risk Assesments: a. Threshold setting : Define threshold below while vechile is considered too close
                    b. Alert Generation: If vechile too close then generate alert
5. Driver Alert System: Display Visual Alert OR Display Audio Alert
6. Integration and Testing: a. Integrate all components into a cohesive applications
                            b. testing app under various conditions to ensure reliability