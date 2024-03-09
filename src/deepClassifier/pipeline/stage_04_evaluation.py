from deepClassifier.config import ConfigurationManager
from deepClassifier.components import Evaluation
from deepClassifier import logger
import os
from dotenv import load_dotenv

STAGE_NAME = "Evaluation"

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = str(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = str(os.getenv("MLFLOW_TRACKING_USERNAME"))
os.environ["MLFLOW_TRACKING_PASSWORD"] = str(os.getenv("MLFLOW_TRACKING_PASSWORD"))


def main():
    config = ConfigurationManager()
    val_config = config.get_validation_config()
    evaluation = Evaluation(val_config)
    evaluation.evaluation()
    evaluation.save_score()
    evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
