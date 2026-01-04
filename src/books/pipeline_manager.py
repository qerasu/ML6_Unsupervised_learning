import numpy as np
from typing import Tuple, Any, Dict, Optional, Self
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import umap


# for FunctionTransformer
def _to_dense(X: Any) -> np.ndarray:
    try:
        return X.toarray()
    except AttributeError:
        return np.asarray(X)


class S21AgePipelineManager:
    def __init__(self, n_iter: int = 10) -> None:
        self.n_iter = int(n_iter)
        self.linear_estimator_ = None
        self.linear_params_ = None
        self.forest_estimator_ = None
        self.forest_params_ = None


    # n_iter must be < n combinations of params
    @staticmethod
    def _n_choices(param_grid: Dict[str, Any]) -> int:
        n = 1

        for v in param_grid.values():
            if isinstance(v, (list, tuple)):
                n *= max(1, len(v))
                
        return max(1, n)


    # returns ridge pipeline
    def _linear_pipeline(self, reduce: Optional[str]) -> Pipeline:
        if reduce is None:
            return Pipeline([
                ("model", Ridge()),
            ])

        key = reduce.lower() if isinstance(reduce, str) else None
        if key == "pca":
            return Pipeline([
                ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
                ("reduce", PCA()),
                ("model", Ridge()),
            ])
        elif key == "umap":
            return Pipeline([
                ("reduce", umap.UMAP()),
                ("model", Ridge()),
            ])
        else:
            raise ValueError("Enter valid reduce (None, PCA or UMAP)")


    # returns forest pipeline
    def _forest_pipeline(self, reduce: Optional[str]) -> Pipeline:
        if reduce is None:
            return Pipeline([
                ("model", RandomForestRegressor(n_jobs=-1)),
            ])

        key = reduce.lower() if isinstance(reduce, str) else None
        if key == "pca":
            return Pipeline([
                ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
                ("reduce", PCA()),
                ("model", RandomForestRegressor(n_jobs=-1)),
            ])
        elif key == "umap":
            return Pipeline([
                ("reduce", umap.UMAP()),
                ("model", RandomForestRegressor(n_jobs=-1)),
            ])
        else:
            raise ValueError("Enter valid reduce (None, PCA or UMAP)")


    # updates params in case of using PCA or UMAP
    def _param_updater(self, params: Dict[str, Any], reduce: Optional[str]) -> Dict[str, Any]:
        key = reduce.lower() if isinstance(reduce, str) else None
        
        if key == "pca":
            params.update({"reduce__n_components": [16, 32]})
        elif key == "umap":
            params.update({
                "reduce__n_components": [16],
                "reduce__n_neighbors": [15],
                "reduce__min_dist": [0.1],
            })

        return params


    # searchs for ridge hyperparamethers usirng _linear_pipeline
    def fit_linear(self, X: csr_matrix, y: np.ndarray, reduce: Optional[str] = None) -> Self:
        params = {"model__alpha": [0.1, 1.0, 10.0]}
        params = self._param_updater(params, reduce)

        n = y.shape[0]
        cv_single = [(np.arange(n), np.arange(n))]
        n_iter = min(self.n_iter, self._n_choices(params))

        search = RandomizedSearchCV(
            self._linear_pipeline(reduce),
            param_distributions=params,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            pre_dispatch=2,
            refit=True,
            verbose=0,
            cv=cv_single,
        )
        search.fit(X, y)

        self.linear_estimator_ = search.best_estimator_
        self.linear_params_ = search.best_params_

        return self


    # searches for ridge hyperparamethers usirng _forest_pipeline
    def fit_forest(self, X: csr_matrix, y: np.ndarray, reduce: Optional[str] = None) -> Self:
        params = {
            "model__n_estimators": [200],
            "model__max_depth": [None, 16],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt"],
        }
        params = self._param_updater(params, reduce)

        n = y.shape[0]
        cv_single = [(np.arange(n), np.arange(n))]
        n_iter = min(self.n_iter, self._n_choices(params))

        search = RandomizedSearchCV(
            self._forest_pipeline(reduce),
            param_distributions=params,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            pre_dispatch=2,
            refit=True,
            verbose=0,
            cv=cv_single,
        )
        search.fit(X, y)

        self.forest_estimator_ = search.best_estimator_
        self.forest_params_ = search.best_params_

        return self


    # gets linear model & it's params
    def get_linear(self) -> Tuple[Pipeline, Dict[str, Any]]:
        if self.linear_estimator_ is None or self.linear_params_ is None:
            raise RuntimeError("Call fit_linear before get_linear")

        return self.linear_estimator_, self.linear_params_


    # gets forest model & it's params
    def get_forest(self) -> Tuple[Pipeline, Dict[str, Any]]:
        if self.forest_estimator_ is None or self.forest_params_ is None:
            raise RuntimeError("Call fit_forest before get_forest")

        return self.forest_estimator_, self.forest_params_


    # shows quality of models
    def evaluate(
        self,
        models: Tuple[Tuple[str, Pipeline], ...],
        X_train: csr_matrix,
        X_val: csr_matrix,
        y: np.ndarray,
    ) -> None:
        if self.linear_estimator_ is None and self.forest_estimator_ is None:
            raise RuntimeError("Call fit_linear/fit_forest before evaluate")

        def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
            return {
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "R2": float(r2_score(y_true, y_pred)),
            }

        for name, est in models:
            y_pred_tr = est.predict(X_train)
            y_pred_val = est.predict(X_val)

            m_tr = _metrics(y, y_pred_tr)
            m_val = _metrics(y, y_pred_val)

            print(f"{name} [train] -> MAE: {m_tr['MAE']:.3f}, RMSE: {m_tr['RMSE']:.3f}, R2: {m_tr['R2']:.3f}")
            print(f"{name} [val]   -> MAE: {m_val['MAE']:.3f}, RMSE: {m_val['RMSE']:.3f}, R2: {m_val['R2']:.3f}")