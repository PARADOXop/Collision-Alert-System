import tensorflow as tf
from pathlib import Path
from deepClassifier.entity import EvaluationConfig
from deepClassifier.utils import save_json
from deepClassifier import logger
import mlflow
import mlflow.keras
from urllib.parse import urlparse

class Evaluation:
    def __init__(self,config:EvaluationConfig):
        self.config = config
        self.score = None

    def _valid_generator(self):
        logger.info("started valid generator")
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.30)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        logger.info(f"loading model from {path}")
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        logger.info("model loaded")
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        logger.info(f"score is {self.score}")

    def save_score(self):
        logger.info("saving score")
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]})
            else:
                mlflow.keras.log_model(self.model, "model")