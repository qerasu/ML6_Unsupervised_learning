from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from data import FaceExpressionDataset, DatasetOutput
from model import FaceExpressionPCA
from translator import EmotionTranslator


class FaceExpressionPipeline:
    # bundle dataset loader, PCA model, and translator
    def __init__(self, dataset: FaceExpressionDataset, model: FaceExpressionPCA) -> None:
        self.dataset = dataset
        self.model = model
        self.translator: Optional[EmotionTranslator] = None
        self.data: Optional[DatasetOutput] = None


    # load data, fit PCA, and compute latent statistics
    def fit(self) -> None:
        self.data = self.dataset.load()
        self.model.fit(self.data.X)
        self.translator = EmotionTranslator(self.model, self.data.label_names)
        self.translator.fit(self.data.X, self.data.y)


    # encode one image vector using the fitted model
    def encode(self, x: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        result = self.translator.encode(x)

        return result


    # decode one latent vector to image space
    def decode(self, z: np.ndarray, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        self._ensure_ready()
        shape = image_shape or self.data.image_shape
        result = self.translator.decode(z, image_shape=shape)

        return result


    # translate an image from source emotion to target emotion
    def translate(
        self,
        x: np.ndarray,
        source_label: int | str,
        target_label: int | str,
        image_shape: Optional[Tuple[int, int]] = None,
        clip: bool = True,
        alpha: float = 1.0,
    ) -> np.ndarray:
        self._ensure_ready()
        shape = image_shape or self.data.image_shape
        result = self.translator.translate(
            x,
            source_label=source_label,
            target_label=target_label,
            image_shape=shape,
            clip=clip,
            alpha=alpha,
        )

        return result


    # raise error if fit has not been called yet
    def _ensure_ready(self) -> None:
        if not self.translator or not self.data:
            raise ValueError("Pipeline is not fitted. Call fit() first.")