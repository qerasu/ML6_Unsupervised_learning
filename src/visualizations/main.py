import gzip
import struct
import umap
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from typing import Optional, Tuple


class S21MnistOperations:
    def __init__(self, images_path: str, labels_path: str) -> None:
        self.images_path: str = images_path
        self.labels_path: str = labels_path
        self.images: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None


    # loads MNIST images from .gz and returns a matrix
    def _load_mnist_images(self, path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(n, rows * cols).astype(np.float32) / 255.0

        return images


    # loads MNIST labels from .gz and returns a vector of labels
    def _load_mnist_labels(self, path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels


    # returns loaded images, loading them from disk if necessary
    def _get_images(self) -> np.ndarray:
        if self.images is None:
            self.images = self._load_mnist_images(self.images_path)

        return self.images



    # returns loaded labels, loading them from disk if necessary
    def _get_labels(self) -> np.ndarray:
        if self.labels is None:
            self.labels = self._load_mnist_labels(self.labels_path)

        return self.labels

    
    # loads all images and labels into attributes and returns them
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.images = self._load_mnist_images(self.images_path)
        self.labels = self._load_mnist_labels(self.labels_path)

        return self.images, self.labels


    # PCA
    def transform_pca(self) -> np.ndarray:
        pca = PCA(n_components=2, random_state=42)

        return pca.fit_transform(self._get_images())
    

    # SVD    
    def transform_svd(self) -> np.ndarray:
        svd = TruncatedSVD(n_components=2, random_state=42)

        return svd.fit_transform(self._get_images())

    # randomized-SVD
    def transform_random_svd(self) -> np.ndarray:
        r_svd = TruncatedSVD(
            n_components=2,
            n_iter=5,
            random_state=42
        )

        return r_svd.fit_transform(self._get_images())


    # TSNE
    def transform_tsne(self) -> np.ndarray:
        X_small = self._get_images()[:5000]
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate='auto',
            init='random',
            random_state=42
        )

        return tsne.fit_transform(X_small)

    
    # UMAP
    def transform_umap(self) -> np.ndarray:
        reducer = umap.UMAP(
            n_components=2
        )

        return reducer.fit_transform(self._get_images())

    # LLE
    def transform_lle(self) -> np.ndarray:
        lle = LocallyLinearEmbedding(
            n_neighbors=10,
            n_components=2,
            method='standard',
            random_state=42
        )

        return lle.fit_transform(self._get_images())