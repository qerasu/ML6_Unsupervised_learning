from __future__ import annotations
from dataclasses import dataclass
from typing import Self
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import pandas as pd


@dataclass
class MatrixSplit:
    train: csr_matrix
    val: csr_matrix
    test: csr_matrix


class S21UserBookMatrixBuilder:
    def __init__(self, ratings_path) -> None:
        ratings = pd.read_csv(ratings_path)
        self.ratings = ratings[ratings["Book-Rating"] > 0].copy()

    # converts ratings to CSR sparse matrix
    def _build_sparse_matrix(self) -> Self:
        ratings = self.ratings

        user_codes = ratings["User-ID"].astype("category")
        book_codes = ratings["ISBN"].astype("category")

        row = user_codes.cat.codes.to_numpy()
        col = book_codes.cat.codes.to_numpy()
        data = ratings["Book-Rating"].to_numpy(dtype="float32")

        self.matrix = coo_matrix(
            (data, (row, col)),
            shape=(
                user_codes.cat.categories.size,
                book_codes.cat.categories.size,
            ),
        ).tocsr()

        return self


    # splits data to train/val/test
    def _split_matrix(
        self,
        train_ratio, val_ratio, test_ratio,
        seed
    ) -> Self:
        self._build_sparse_matrix()
        matrix = self.matrix

        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
        if not np.isclose(ratios.sum(), 1.0):
            raise ValueError("Sum of ratios must be 1")

        ratings_coo = matrix.tocoo()
        nnz = ratings_coo.nnz
        indices = np.arange(nnz)

        if seed is not None:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

        train_end = int(ratios[0] * nnz)
        val_end = train_end + int(ratios[1] * nnz)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        def build(idx: np.ndarray) -> csr_matrix:
            if idx.size == 0:
                return csr_matrix(ratings_coo.shape, dtype=ratings_coo.data.dtype)
            return coo_matrix(
                (ratings_coo.data[idx], (ratings_coo.row[idx], ratings_coo.col[idx])),
                shape=ratings_coo.shape,
            ).tocsr()

        self.matrix_split = MatrixSplit(
            train=build(train_idx),
            val=build(val_idx),
            test=build(test_idx),
        )

        return self


    # public method calling private methods building and splitting matrix
    def build_split(
        self,
        ratios,
        seed = None
    ) -> MatrixSplit:
        if len(ratios) != 3 or any(r < 0 for r in ratios):
            raise ValueError("Enter valid ratios") 

        train_ratio, val_ratio, test_ratio = ratios
        self._split_matrix(train_ratio, val_ratio, test_ratio, seed)

        split = self.matrix_split

        return split