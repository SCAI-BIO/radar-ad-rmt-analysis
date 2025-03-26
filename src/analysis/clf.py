# Imports
import logging
import random
import sys
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
import typer
from optuna.exceptions import ExperimentalWarning
from optuna.integration import OptunaSearchCV
from optuna.samplers import TPESampler
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, shuffle
from xgboost import XGBClassifier

from src.definitions import CLASSIFIER, NEW_GROUPS_INVERSE, POT_CONFOUDNERS, SCALE_EXCLUDE, NEW_GROUPS, COMPARISONS
from src.utils.set_all_seeds import set_all_seeds

# Fixed random seed
np.random.seed(42)
check_random_state(42)

# globals
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)  # or logging.ERROR to suppress even more
logger = logging.getLogger()
warnings.filterwarnings("ignore", category=ExperimentalWarning)


# classes
class MissingSampleException(Exception):
    pass


class Trainer(ABC):
    """
    Base class for all trainers.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        groups: np.ndarray,
        params: Dict = {},
        folds: int = 10,
        seed: int = 220906,
        cores: int = 4,
        single_observations: bool = True,
        impute_values: bool = False,
        imputer: str = "knn",
        data_name: str = "data",
        db_storage: str = "sqlite:///optuna.db",
    ) -> None:
        self.data = data
        self.groups = groups
        self.params = params
        self.folds = folds
        self.seed = seed
        self.cores = cores
        self.single_observations = single_observations
        self.strater = StratifiedKFold if single_observations else StratifiedGroupKFold
        self.type = "default"
        self.data_name = data_name
        self.db_storage = db_storage
        self.impute_values = impute_values
        self.imputer = imputer

        set_all_seeds(seed)

    def get_imputer(
        self, imp_type="iterative_rf"
    ) -> Union[IterativeImputer, KNNImputer, SimpleImputer]:
        """
        Returns an imputer object based on the specified imputation type.

        Parameters:
            imp_type (str): The type of imputation to use. Options are:
                - "iterative_rf": Uses an IterativeImputer with a RandomForestRegressor as the estimator.
                - "knn": Uses a KNNImputer with 5 neighbors.
                - "mean": Uses a SimpleImputer with the "mean" strategy.
                - "median": Uses a SimpleImputer with the "median" strategy.
                Default is "iterative_rf".

        Returns:
            Union[IterativeImputer, KNNImputer, SimpleImputer]: The configured imputer object.

        Raises:
            NotImplementedError: If the specified `imp_type` is not recognized.
        """
        if imp_type == "iterative_rf":
            est = RandomForestRegressor(
                n_estimators=4,
                max_depth=10,
                bootstrap=True,
                max_samples=0.5,
                n_jobs=self.cores,
                random_state=self.seed,
            )
            return IterativeImputer(
                estimator=est,
                random_state=self.seed,
                max_iter=10,
                tol=1e-1,
                n_nearest_features=5,
            )
        elif imp_type == "knn":
            return KNNImputer(n_neighbors=5)
        elif imp_type == "mean":
            return SimpleImputer(strategy="mean")
        elif imp_type == "median":
            return SimpleImputer(strategy="median")
        else:
            raise NotImplementedError

    @abstractmethod
    def setup_clf(self, is_train: bool = False) -> CLASSIFIER:  # type: ignore  # type: ignore
        """
        Setup the classifier with current parameters

        Parameters:
            is_train: Flag if classifier is used for final training

        Returns:
            Classifier
        """
        pass

    @abstractmethod
    def hpo_params(self) -> Dict:
        """
        Return hyperparameter search space for current classifier.

        Parameters:
            None

        Returns:
            Dict: Hyperparameter search space.
        """
        pass

    @abstractmethod
    def default_parameters(self) -> Dict:
        """
        Return default hyperparameter values for respective classifier.

        Parameters:
            None

        Returns:
            Dict: Default hyperparameter values.
        """
        pass

    @staticmethod
    def merge_parameters(
        defaults: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge the default parameters with user-provided parameters.

        Parameters:
            defaults (Dict[str, Any]): The default parameter dictionary.
            params (Dict[str, Any]): User-provided parameters to override defaults.

        Returns:
            Dict[str, Any]: A dictionary containing the merged parameters,
                            where user-provided parameters take precedence over defaults.
        """
        for key, value in params.items():
            defaults[key] = value

        logger.info("Modified params")
        logger.info(defaults)
        return defaults

    def stratify_data(
        self,
    ) -> Generator[Tuple[Tuple[Any, Any], Tuple[Any, Any]], None, None]:
        """
        Stratify the data based on the groups and the number of folds.

        Yields:
            Generator[Tuple[Tuple[Any, Any], Tuple[Any, Any]], None, None]: Stratified train-test splits.
        """
        sss = self.strater(n_splits=self.folds, shuffle=True, random_state=self.seed)
        X = self.data.to_numpy()
        Y = np.array(self.groups)
        for train, test in sss.split(
            X, Y, list(self.data.index) if self.single_observations else None
        ):
            yield (X[train, :], Y[train]), (X[test, :], Y[test])

    def train(self) -> pd.DataFrame:
        """
        Train the classifier using stratified cross-validation.

        Returns:
            pd.DataFrame: The results of the training.
        """
        scores = []
        for i, ((x_train, y_train), (x_test, y_test)) in enumerate(
            self.stratify_data()
        ):
            clf = self.setup_clf(is_train=True)
            clf.fit(x_train, y_train)
            y_predictions = clf.predict(x_test)
            metrics = classification_report(
                y_test,
                y_predictions,
                zero_division=0,
                output_dict=True,  # type: ignore
            )
            scores.append(pd.DataFrame(metrics))
            scores[-1]["split"] = i

        results = pd.concat(scores)
        return results

    @staticmethod
    def get_subset(
        x: pd.DataFrame,
        y: np.ndarray,
        ids: np.ndarray,
        comparison: Tuple[int, int],
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Get a subset of the data based on the comparison.

        Parameters:
            x (pd.DataFrame): The input data.
            y (np.ndarray): The target labels.
            ids (np.ndarray): The patient IDs.
            comparison (Tuple[int, int]): The comparison to use indicated with the group IDs.
        """
        match comparison:
            case (0, -1):
                # HC vs. All
                sub_y = y
                sub_x = x
                sub_ids = ids
            case (0, 4):
                # HC vs. MildAD+ProAD
                named_groups = ["ProAD", "MildAD"]
                group_ids = [0] + [NEW_GROUPS_INVERSE[x] - 1 for x in named_groups]
                comparison_index = [True if val in group_ids else False for val in y]
                sub_y = y[comparison_index]
                sub_x = x.loc[comparison_index, :]
                sub_ids = ids[comparison_index]
            case _:
                comparison_index = np.isin(y, comparison)
                sub_y = y[comparison_index]
                sub_x = x.loc[comparison_index, :]
                sub_ids = ids[comparison_index]

        sub_y = np.where(sub_y == comparison[0], 0, 1)
        return sub_x, sub_y, sub_ids

    def repeated_ncv(
        self,
        trials: int = 10,
        repeats: int = 10,
        comparisons: List[Tuple[int, int]] = COMPARISONS,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform repeated stratified cross-validation on the classifier.

        Args:
            trials (int): The number of trials to perform.
            repeats (int): The number of times to repeat the cross-validation.
            comparisons (List[Tuple[int, int]]): The comparisons to make.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The results of the training.
        """

        random.seed(self.seed)

        # Generate random seeds for the number of repeats
        random_seeds: Set[int] = set()
        while len(random_seeds) < repeats:
            random_seeds.add(random.randint(0, 100_000))

        overall_metrics = []
        overall_shap_values = []
        overall_test_shap_values = []

        for seed in random_seeds:
            ncv_df, shap_values, test_shap_values = self.ncv(trials, comparisons, seed)
            overall_metrics.append(ncv_df)
            overall_shap_values.append(shap_values)
            overall_test_shap_values.append(test_shap_values)

        overall_df = pd.concat(overall_metrics)
        overall_shap_df = pd.concat(overall_shap_values)
        overall_test_shap_df = pd.concat(overall_test_shap_values)

        return overall_df, overall_shap_df, overall_test_shap_df

    def fit_models(
        self,
        trials: int,
        comparisons: List[Tuple[int, int]],
        seed: int,
        excluded_features: Optional[List[str]] = None,
    ) -> List[ClassifierMixin]:
        """
        Run the inner procedure per comparison and fit the final models

        Parameters:
            trials (int): Number of trials to run for each comparison.
            comparisons (List[Tuple[int, int]]): List of tuples representing the comparisons.
            seed (int): Seed for random number generator.
            excluded_features (Optional[List[str]]): List of features to exclude from data.

        Returns:
            List[ClassifierMixin]: List of fitted classifiers.
        """
        logger.info(f"Start fitting models with seed {seed}")

        # Exclude features from data that are not needed
        if excluded_features:
            logger.info(f"Excluded features: {excluded_features}")
            available_features = self.data.columns
            dropable_features = [
                x for x in excluded_features if x in available_features
            ]
            if dropable_features:
                logger.info(f"Dropping features: {dropable_features}")
                self.data.drop(columns=dropable_features, inplace=True)

        # Shuffle data before fitting
        assert isinstance(seed, int), "Seed must be an integer"
        assert isinstance(self.data, pd.DataFrame), "Data must be a pandas DataFrame"
        assert self.data.shape[0] > 0, "Data must not be empty"

        shuffled_x, shuffled_y = shuffle(self.data, self.groups, random_state=seed)  # type: ignore

        # Create the classifier
        base_clf = self.setup_clf()

        inner_cv = self.strater(n_splits=self.folds, shuffle=True, random_state=seed)
        patient_ids = np.array(list(shuffled_x.index))  # type: ignore

        group_map = {}
        for pid in patient_ids:
            if pid not in group_map:
                group_map[pid] = len(group_map)
        group_ids = np.array(list(map(group_map.get, patient_ids)))

        models = []

        for comparison in comparisons:
            logger.info(
                f"Start comparison {NEW_GROUPS[comparison[0] + 1]} vs. {NEW_GROUPS[comparison[1] + 1]}"
            )
            sub_x, sub_y, sub_ids = self.get_subset(
                shuffled_x,  # type: ignore
                shuffled_y,  # type: ignore
                group_ids,
                comparison,
            )
            try:
                sub_x.set_index("record_id", inplace=True)
            except:  # noqa: E722
                logger.debug("Could not set index")
                pass
            inner_split = inner_cv.split(
                sub_x,
                sub_y,
                groups=sub_ids if not self.single_observations else None,
            )
            optuna_search = self.inner_procedure(
                trials,
                deepcopy(base_clf),
                inner_split,
                sub_x,
                sub_y,
                sub_ids,
                comparison,
                outer_fold="final" if not excluded_features else "final_reduced",
                db_storage=self.db_storage,
                seed=seed,
            )

            est = optuna_search.best_estimator_
            models.append(est)

        return models

    def ncv(
        self,
        trials: int,
        comparisons: List[Tuple[int, int]],
        seed: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform nested cross-validation (NCV) on the dataset.

        Parameters:
            trials (int): Number of trials for hyperparameter optimization.
            comparisons (List[Tuple[int, int]]): List of tuples representing feature comparisons.
            seed (int): Random seed for reproducibility.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames containing NCV metrics, SHAP values, and test SHAP values.
        """

        logger.info(f"Start NCV with seed {seed}")

        # shuffle data and groups
        shuffled_x, shuffled_y = shuffle(self.data, self.groups, random_state=seed)  # type: ignore

        # Get classifier instance
        base_clf = self.setup_clf()

        # Set up CV splitter
        inner_cv = self.strater(n_splits=self.folds, shuffle=True, random_state=seed)
        outer_cv = self.strater(n_splits=self.folds, shuffle=True, random_state=seed)
        patient_ids = np.array(list(shuffled_x.index))  # type: ignore

        group_map = {}
        for pid in patient_ids:
            if pid not in group_map:
                group_map[pid] = len(group_map)

        ncv_metrics = []
        shap_values_list = []
        test_shap_values_list = []

        for i, (outer_train_idx, outer_test_idx) in enumerate(
            outer_cv.split(
                shuffled_x,
                shuffled_y,
            )
        ):
            logger.info(f"Start outer fold {i + 1}")
            outer_train_x = shuffled_x.iloc[outer_train_idx, :]  # type: ignore
            outer_train_y = shuffled_y[outer_train_idx]
            outer_patient_train_ids = patient_ids[outer_train_idx]

            outer_test_x = shuffled_x.iloc[outer_test_idx, :]  # type: ignore
            outer_test_y = shuffled_y[outer_test_idx]
            outer_patient_test_ids = patient_ids[outer_test_idx]

            split_metrics = []

            for comparison in comparisons:
                a, b = comparison
                a += 1
                b += 1
                logger.info(f"Start comparison {NEW_GROUPS[a]} vs. {NEW_GROUPS[b]}")
                sub_x, sub_y, sub_ids = self.get_subset(
                    outer_train_x,
                    outer_train_y,
                    outer_patient_train_ids,
                    comparison,
                )
                try:
                    sub_x.set_index("record_id", inplace=True)
                except:  # noqa: E722
                    logger.debug("Could not set index")
                    pass
                inner_split = inner_cv.split(
                    sub_x,
                    sub_y,
                    groups=sub_ids if not self.single_observations else None,
                )
                try:
                    optuna_search = self.inner_procedure(
                        trials,
                        deepcopy(base_clf),
                        inner_split,
                        sub_x,
                        sub_y,
                        sub_ids,
                        comparison,
                        outer_fold=str(i),
                        db_storage=self.db_storage,
                        seed=seed,
                    )

                    trainsub_x, _, trainsub_ids = self.get_subset(
                        outer_train_x,
                        outer_train_y,
                        outer_patient_train_ids,
                        comparison,
                    )

                    testsub_x, testsub_y, testsub_ids = self.get_subset(
                        outer_test_x,
                        outer_test_y,
                        outer_patient_test_ids,
                        comparison,
                    )

                    est = optuna_search.best_estimator_

                    # Compute shap values
                    # if we have a sklearn pipeline, we have to extract each component
                    # and pass it to shap
                    if isinstance(est, Pipeline):
                        scaler = (
                            est.named_steps["scale"]
                            if "scale" in est.named_steps
                            else None
                        )
                        imputer = (
                            est.named_steps["imputer"]
                            if "imputer" in est.named_steps
                            else None
                        )
                        # extract the model from the pipeline
                        model_name = [
                            x
                            for x in est.named_steps.keys()
                            if x not in ["scale", "imputer"]
                        ][0]
                        model = est.named_steps[model_name]

                        (
                            _,
                            train_shap_values,
                        ) = self.calculate_shap_values(
                            data_x=trainsub_x,
                            scaler=scaler,
                            imputer=imputer,
                            model=model,
                        )

                        (
                            _,
                            test_shap_values,
                        ) = self.calculate_shap_values(
                            data_x=testsub_x,
                            scaler=scaler,
                            imputer=imputer,
                            model=model,
                        )

                    else:
                        raise NotImplementedError

                    # Store SHAP values along with fold and comparison info
                    train_shap_values_df = pd.DataFrame(
                        train_shap_values, columns=sub_x.columns
                    )
                    train_shap_values_df["split"] = i
                    train_shap_values_df["seed"] = seed
                    train_shap_values_df["record_id"] = trainsub_ids
                    train_shap_values_df["comparison"] = (
                        f"{str(comparison[0])} vs. {str(comparison[1])}"
                    )
                    shap_values_list.append(train_shap_values_df)

                    test_shap_values_df = pd.DataFrame(
                        test_shap_values, columns=testsub_x.columns
                    )
                    test_shap_values_df["split"] = i
                    test_shap_values_df["seed"] = seed
                    test_shap_values_df["record_id"] = testsub_ids
                    test_shap_values_df["comparison"] = (
                        f"{str(comparison[0])} vs. {str(comparison[1])}"
                    )
                    test_shap_values_list.append(test_shap_values_df)

                    preds = est.predict(testsub_x)
                    try:
                        auroc = roc_auc_score(testsub_y, preds)
                        precision, recall, _ = precision_recall_curve(testsub_y, preds)
                        aupr = auc(recall, precision)
                    except ValueError:
                        auroc = None
                        aupr = None
                    logger.info(f"AUROC: {auroc}")
                    logger.info(f"AUPR: {aupr}")

                    metrics = {
                        "comparison": f"{str(comparison[0])} vs. {str(comparison[1])}",
                        "auroc": auroc,
                        "aupr": aupr,
                        "split": i,
                        "seed": seed,
                    }
                except MissingSampleException:
                    metrics = {
                        "comparison": f"{str(comparison[0])} vs. {str(comparison[1])}",
                        "auroc": None,
                        "aupr": None,
                        "split": i,
                        "seed": seed,
                    }
                split_metrics.append(metrics)

            split_df = pd.DataFrame(split_metrics)

            ncv_metrics.append(split_df)

        # Combine SHAP values from all folds and comparisons into one DataFrame
        all_shap_values = pd.concat(shap_values_list, ignore_index=True)
        all_test_shap_values = pd.concat(test_shap_values_list, ignore_index=True)

        ncv_df = pd.concat(ncv_metrics)
        return ncv_df, all_shap_values, all_test_shap_values

    @staticmethod
    def calculate_shap_values(
        model: Union[LogisticRegression, RandomForestClassifier, XGBClassifier],
        data_x: Union[pd.DataFrame, np.ndarray],
        scaler: Optional[StandardScaler] = None,
        imputer: Optional[KNNImputer] = None,
    ) -> Tuple[shap.Explainer, np.ndarray]:
        """
        Calculate SHAP values for a given model and data.

        Args:
            model: The trained model.
            data_x: The input data.
            scaler: Optional scaler for data normalization.
            imputer: Optional imputer for missing values.

        Returns:
            A tuple containing the SHAP explainer and SHAP values.
        """

        logger.info("Calculate SHAP values")
        logger.info(f"Shape of data: {data_x.shape}")  # type: ignore

        scaled_data = scaler.transform(data_x) if scaler else data_x
        logger.info(f"Shape of scaled data: {scaled_data.shape}")  # type: ignore

        imputed_data = imputer.transform(scaled_data) if imputer else scaled_data
        logger.info(f"Shape of imputed data: {imputed_data.shape}")  # type: ignore

        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, imputed_data)
            shap_values = explainer.shap_values(imputed_data)

        elif isinstance(model, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model, imputed_data)
            shap_values = explainer.shap_values(imputed_data, check_additivity=False)
            if shap_values.ndim == 3:
                # For RF, the shap values are a 3D array, where the last dimension
                # corresponds to the class label. We are only interested in the last
                # class label, so we select it here.
                shap_values = shap_values[:, :, -1]

        else:
            raise NotImplementedError(
                f"Model type {type(model)} not supported for SHAP calculation."
            )

        logger.info(f"Shape of SHAP values: {shap_values.shape}")  # type: ignore

        return explainer, shap_values

    def inner_procedure(
        self,
        trials: int,
        clf: Union[XGBClassifier, RandomForestClassifier, LogisticRegression, Pipeline],
        split_gen: Generator,
        x_data: Union[pd.DataFrame, np.ndarray],
        y_data: np.ndarray,
        pat_ids: np.ndarray,
        comparison: Tuple[int, int],
        outer_fold: str,
        db_storage: str,
        seed: int,
    ):
        """
        Perform the inner optimization & fit procedure

        Parameters:
            trials (int): Number of trials to run
            clf (object): Classifier object
            split_gen (generator): Generator for train/test splits
            x_data (array): Feature data
            y_data (array): Label data
            pat_ids (array): Patient IDs
            comparison (tuple): Comparison tuple
            outer_fold (int): Outer fold number
            db_storage (str): Database storage string
            seed (int): Random seed

        Returns:
            study (object): Optuna study object
        """

        start = time.time()
        study = optuna.create_study(
            storage=db_storage,
            study_name=f"{self.type}_{outer_fold}_{comparison[0]}-{comparison[1]}_{self.data_name}_{str(seed)}",
            direction="maximize",
            sampler=TPESampler(seed=202209),
            load_if_exists=True,
        )
        study._storage._backend.engine.dispose()
        # Determine how many trials are left
        complete_trials = len(
            [x for x in study.trials if x.state == optuna.trial.TrialState.COMPLETE]
        )
        trials_to_do = trials - complete_trials
        if trials_to_do <= 0:
            trials_to_do = 0
        logger.info(f"Start inner procedure with {trials_to_do} trials")

        optuna_search = OptunaSearchCV(
            estimator=clf,
            param_distributions=self.hpo_params(),
            cv=split_gen,
            scoring="roc_auc",
            refit=True,
            random_state=self.seed,
            study=study,
            n_trials=trials_to_do,
        )
        optuna_search.fit(
            x_data,
            y_data,
            groups=pat_ids,
        )
        logger.info(f"Inner procedure took {time.time() - start} seconds")
        return optuna_search


class XgbTrainer(Trainer):
    def __init__(
        self,
        data: pd.DataFrame,
        groups: np.ndarray,
        params: Dict = {},
        folds: int = 10,
        seed: int = 220906,
        cores: int = 4,
        only_confounders: bool = False,
        single_observation_data: bool = True,
        impute_values: bool = False,
        imputer: str = "knn",
        data_name: str = "data",
        db_storage: str = "sqlite:///optuna.db",
    ) -> None:
        super().__init__(
            data=data,
            groups=groups,
            params=params,
            folds=folds,
            seed=seed,
            cores=cores,
            single_observations=single_observation_data,
            impute_values=impute_values,
            imputer=imputer,
            data_name=data_name,
            db_storage=db_storage,
        )
        self.type = "xgb"

    def default_parameters(self) -> Dict:
        """
        Return default hyperparameter values for respective classifier.

        Parameters:
            None

        Returns:
            Dict: Default hyperparameter values.
        """
        parameters = {
            "n_estimators": 100,
            "random_state": self.seed,
            "nthread": 4,
        }
        return parameters

    def setup_clf(self, is_train: bool = False) -> CLASSIFIER:  # type: ignore
        """
        Setup the classifier with current parameters

        Parameters:
            is_train: Flag if classifier is used for final training

        Returns:
            Classifier
        """
        default = self.default_parameters()
        params = self.merge_parameters(default, self.params)
        if is_train:
            params["nthread"] = 1
            params["deterministic"] = True
        clf = XGBClassifier(**params)
        if not self.impute_values:
            return clf
        else:
            logger.warning("Use implicit imputation")
            return Pipeline(
                steps=[
                    ("xgb", clf),
                ]
            )

    def hpo_params(self) -> Dict:
        """
        Return hyperparameter search space for current classifier.

        Parameters:
            None

        Returns:
            Dict: Hyperparameter search space.
        """
        parameters = {
            "eta": optuna.distributions.FloatDistribution(0.01, 0.7, log=True),
            "gamma": optuna.distributions.FloatDistribution(0, 0.5, step=0.1),
            "max_depth": optuna.distributions.IntDistribution(1, 22, step=3),
            "n_estimators": optuna.distributions.IntDistribution(50, 400, step=25),
        }
        if self.impute_values:
            parameters = {f"xgb__{k}": v for k, v in parameters.items()}

        return parameters


class RfTrainer(Trainer):
    def __init__(
        self,
        data: pd.DataFrame,
        groups: np.ndarray,
        params: Dict = {},
        folds: int = 10,
        seed: int = 220906,
        cores: int = 4,
        only_confounders: bool = False,
        single_observation_data: bool = True,
        impute_values: bool = False,
        imputer: str = "knn",
        data_name: str = "data",
        db_storage: str = "sqlite:///optuna.db",
    ) -> None:
        super().__init__(
            data=data,
            groups=groups,
            params=params,
            folds=folds,
            seed=seed,
            cores=cores,
            single_observations=single_observation_data,
            impute_values=impute_values,
            imputer=imputer,
            data_name=data_name,
            db_storage=db_storage,
        )
        self.type = "rf"

    def default_parameters(self) -> Dict:
        """
        Return default hyperparameter values for respective classifier.

        Parameters:
            None

        Returns:
            Dict: Default hyperparameter values.
        """
        parameters = {
            "n_estimators": 1000,
            "random_state": self.seed,
            "n_jobs": 8,
        }
        return parameters

    def setup_clf(self, is_train: bool = False) -> CLASSIFIER:  # type: ignore
        """
        Setup the classifier with current parameters

        Parameters:
            is_train: Flag if classifier is used for final training

        Returns:
            Classifier
        """
        default = self.default_parameters()
        if is_train:
            default["n_jobs"] = 1
        params = self.merge_parameters(default, self.params)
        numeric_features = [
            col for col in self.data.columns if col not in SCALE_EXCLUDE
        ]
        clf = RandomForestClassifier(**params)
        if not self.impute_values:
            return clf
        else:
            return Pipeline(
                steps=[
                    (
                        "scale",
                        ColumnTransformer(
                            [("num", StandardScaler(), numeric_features)],
                            remainder="passthrough",
                        ),
                    ),
                    ("imputer", self.get_imputer(self.imputer)),
                    ("rf", clf),
                ]
            )

    def hpo_params(self) -> Dict:
        """
        Return hyperparameter search space for current classifier.

        Parameters:
            None

        Returns:
            Dict: Hyperparameter search space.
        """
        parameters = {
            "n_estimators": optuna.distributions.IntDistribution(600, 1400, step=200),
            "max_depth": optuna.distributions.IntDistribution(10, 50, step=10),
            "min_samples_leaf": optuna.distributions.CategoricalDistribution([1, 2, 4]),
            "min_samples_split": optuna.distributions.CategoricalDistribution(
                [2, 5, 10]
            ),
        }
        if self.impute_values:
            parameters = {f"rf__{k}": v for k, v in parameters.items()}

        return parameters


class CartTrainer(Trainer):
    def __init__(
        self,
        data: pd.DataFrame,
        groups: np.ndarray,
        params: Dict = {},
        folds: int = 10,
        seed: int = 220906,
        cores: int = 4,
        only_confounders: bool = False,
        single_observation_data: bool = True,
        impute_values: bool = False,
        imputer: str = "knn",
        data_name: str = "data",
        db_storage: str = "sqlite:///optuna.db",
    ) -> None:
        super().__init__(
            data=data,
            groups=groups,
            params=params,
            folds=folds,
            seed=seed,
            cores=cores,
            single_observations=single_observation_data,
            impute_values=impute_values,
            imputer=imputer,
            data_name=data_name,
            db_storage=db_storage,
        )
        self.type = "cart"

    def default_parameters(self) -> Dict:
        """
        Return default hyperparameter values for respective classifier.

        Parameters:
            None

        Returns:
            Dict: Default hyperparameter values.
        """
        parameters = {
            "random_state": self.seed,
        }
        return parameters

    def setup_clf(self, is_train: bool = False) -> CLASSIFIER:  # type: ignore
        """
        Setup the classifier with current parameters

        Parameters:
            is_train: Flag if classifier is used for final training

        Returns:
            Classifier
        """
        default = self.default_parameters()
        params = self.merge_parameters(default, self.params)
        numeric_features = [
            col for col in self.data.columns if col not in SCALE_EXCLUDE
        ]
        clf = RandomForestClassifier(
            n_estimators=1, bootstrap=False, max_features=None, **params
        )
        if not self.impute_values:
            return clf
        else:
            return Pipeline(
                steps=[
                    (
                        "scale",
                        ColumnTransformer(
                            [("num", StandardScaler(), numeric_features)],
                            remainder="passthrough",
                        ),
                    ),
                    ("imputer", self.get_imputer(self.imputer)),
                    ("cart", clf),
                ]
            )

    def hpo_params(self) -> Dict:
        """
        Return hyperparameter search space for current classifier.

        Parameters:
            None

        Returns:
            Dict: Hyperparameter search space.
        """
        parameters = {
            "max_depth": optuna.distributions.IntDistribution(10, 50, step=10),
            "min_samples_leaf": optuna.distributions.CategoricalDistribution([1, 2, 4]),
            "min_samples_split": optuna.distributions.CategoricalDistribution(
                [2, 5, 10]
            ),
        }
        if self.impute_values:
            parameters = {f"cart__{k}": v for k, v in parameters.items()}

        return parameters


class LogRegTrainer(Trainer):
    def __init__(
        self,
        data: pd.DataFrame,
        groups: np.ndarray,
        params: Dict = {},
        folds: int = 10,
        seed: int = 220906,
        cores: int = 4,
        only_confounders: bool = False,
        single_observation_data: bool = True,
        impute_values: bool = False,
        imputer: str = "knn",
        data_name: str = "data",
        db_storage: str = "sqlite:///optuna.db",
    ) -> None:
        super().__init__(
            data=data,
            groups=groups,
            params=params,
            folds=folds,
            seed=seed,
            cores=cores,
            single_observations=single_observation_data,
            impute_values=impute_values,
            imputer=imputer,
            data_name=data_name,
            db_storage=db_storage,
        )
        self.orig_data = self.data
        self.data = self.data
        self.data.columns = self.orig_data.columns
        self.type = "lr"

    def default_parameters(self) -> Dict:
        """
        Return default hyperparameter values for respective classifier.

        Parameters:
            None

        Returns:
            Dict: Default hyperparameter values.
        """
        return {
            "max_iter": 5000,
            "random_state": self.seed,
            "penalty": "elasticnet",
            "solver": "saga",
            "n_jobs": 1,
        }

    def setup_clf(self, is_train: bool = False) -> CLASSIFIER:  # type: ignore
        """
        Setup the classifier with current parameters

        Parameters:
            is_train: Flag if classifier is used for final training

        Returns:
            Classifier
        """
        default = self.default_parameters()
        params = self.merge_parameters(default, self.params)
        numeric_features = [
            col for col in self.data.columns if col not in SCALE_EXCLUDE
        ]
        if self.impute_values:
            steps = [
                (
                    "scale",
                    ColumnTransformer(
                        [("num", StandardScaler(), numeric_features)],
                        remainder="passthrough",
                    ),
                ),
                ("imputer", self.get_imputer(self.imputer)),
                ("lr", LogisticRegression(**params)),
            ]
        else:
            steps = [
                (
                    "scale",
                    ColumnTransformer(
                        [("num", StandardScaler(), numeric_features)],
                        remainder="passthrough",
                    ),
                ),
                ("lr", LogisticRegression(**params)),
            ]
        pipe = Pipeline(steps=steps)
        return pipe

    def hpo_params(self) -> Dict:
        """
        Return hyperparameter search space for current classifier.

        Parameters:
            None

        Returns:
            Dict: Hyperparameter search space.
        """
        params = {
            "lr__C": optuna.distributions.FloatDistribution(0.01, 2, log=True),
            "lr__l1_ratio": optuna.distributions.FloatDistribution(
                0.01, 0.99, log=True
            ),
        }
        return params


# Executable function

app = typer.Typer()
ncv_app = typer.Typer()
fit_app = typer.Typer()

app.add_typer(ncv_app, name="ncv", help="Nested cross-validation")
app.add_typer(fit_app, name="fit", help="Fit models")


def prepare_data(
    x_data: Path,
    y_data: Path,
    only_confounder: bool,
    optional_data: Optional[Path] = None,
    single_observation_data: bool = True,
    keepna: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare data for analysis.

    Reads CSV files, selects appropriate columns, and merges data.

    Parameters:
        x_data (Path): Path to the CSV file containing the features.
        y_data (Path): Path to the CSV file containing the target variable.
        only_confounder (bool): Whether to select only the confounders.
        optional_data (Optional[Path]): Path to the CSV file containing optional data.
        single_observation_data (bool): Whether to keep only one observation per record.
        keepna (bool): Whether to keep rows with missing values.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: The prepared data and the target variable.
    """
    xdf = pd.read_csv(x_data, index_col="record_id")
    gdf = pd.read_csv(y_data)
    if only_confounder:
        selection = sorted([x for x in POT_CONFOUDNERS if x in xdf.columns])
        xdf = xdf.loc[:, selection]  # type: ignore

    if optional_data is not None:
        # Potentially merge with other file
        opt_data = pd.read_csv(optional_data, index_col="record_id")
        to_drop = [x for x in opt_data.columns if x != "record_id" and x in xdf.columns]
        opt_data.drop(labels=to_drop, axis=1, inplace=True)

        xdf = pd.merge(
            xdf,
            gdf.set_index("record_id"),
            left_index=True,
            right_index=True,
            how="left",
        )
        xdf = pd.merge(xdf, opt_data, left_index=True, right_index=True, how="left")

        if not keepna:
            # Remove rows with missing values
            rows_to_keep = list(xdf.notna().all(axis=1))
            logger.warning(
                f"{len(rows_to_keep)} rows had to be removed due to missing values"
            )
            xdf = xdf.loc[rows_to_keep, :]

        gdf = xdf.reset_index().loc[:, ["record_id", "group"]]
        xdf.drop("group", axis=1, inplace=True)

    assert "group" in gdf.columns, "Missing 'group' column"
    groups = gdf.group.to_numpy().astype(int)
    if min(groups) > 0:
        groups = groups - np.min(groups)
    return xdf.sort_index(axis=1), groups


def prepare_logging():
    """"""
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.info("Start logging")


@ncv_app.command()
def rf(
    x_data: Path,
    y_data: Path,
    output_file: Path,
    train_shap_file: Path,
    test_shap_file: Path,
    folds: int = 10,
    cores: int = 4,
    trials: int = 50,
    repeats: int = 10,
    additional_data: Optional[Path] = None,
    only_confounders: bool = False,
    single_observation_data: str = "n",
    repeat: Optional[int] = None,
    impute_values: bool = False,
    db_storage: str = "sqlite:///optuna.db",
    data_name: str = "data",
):
    prepare_logging()
    config = dict()
    random.seed(repeat)
    np.random.seed(repeat)

    # Flag if only one obersvation should be available per participant
    reduce_df = True if single_observation_data == "y" else False

    # Prepare data for
    xdf, groups = prepare_data(
        x_data,
        y_data,
        only_confounders,
        single_observation_data=reduce_df,
        optional_data=additional_data,
        keepna=impute_values,
    )
    trainer = RfTrainer(
        xdf,
        groups,
        config,
        folds=folds,
        cores=cores,
        impute_values=impute_values,
        db_storage=db_storage,
        data_name=f"{data_name}_{only_confounders}_{'A' if additional_data else 'N'}",
    )
    comparisons_to_run = COMPARISONS
    if "altoida" in data_name:
        mild_ad_id = NEW_GROUPS_INVERSE["MildAD"] - 1
        proad_mildad_id = NEW_GROUPS_INVERSE["ProAD+MildAD"] - 1
        comparisons_to_run = list(
            (a, b)
            for (a, b) in COMPARISONS
            if (a != mild_ad_id and b != mild_ad_id)
            or (a != proad_mildad_id and b != proad_mildad_id)
        )
    if repeat is not None:
        results, shap_values, test_shap_values = trainer.ncv(
            trials, comparisons_to_run, seed=repeat
        )
    else:
        results, shap_values, test_shap_values = trainer.repeated_ncv(
            trials, repeats=repeats
        )
    results["model"] = f"rf{'_base' if only_confounders else ''}"
    shap_values["model"] = f"rf{'_base' if only_confounders else ''}"
    test_shap_values["model"] = f"rf{'_base' if only_confounders else ''}"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(output_file)
    shap_values.to_csv(train_shap_file)
    test_shap_values.to_csv(test_shap_file)


@ncv_app.command()
def cart(
    x_data: Path,
    y_data: Path,
    output_file: Path,
    train_shap_file: Path,
    test_shap_file: Path,
    folds: int = 10,
    cores: int = 4,
    trials: int = 50,
    repeats: int = 10,
    additional_data: Optional[Path] = None,
    only_confounders: bool = False,
    single_observation_data: str = "n",
    repeat: Optional[int] = None,
    impute_values: bool = False,
    db_storage: str = "sqlite:///optuna.db",
    data_name: str = "data",
):
    prepare_logging()
    config = dict()
    random.seed(repeat)
    np.random.seed(repeat)

    # Flag if only one obersvation should be available per participant
    reduce_df = True if single_observation_data == "y" else False

    # Prepare data for
    xdf, groups = prepare_data(
        x_data,
        y_data,
        only_confounders,
        single_observation_data=reduce_df,
        optional_data=additional_data,
        keepna=impute_values,
    )
    trainer = CartTrainer(
        xdf,
        groups,
        config,
        folds=folds,
        cores=cores,
        impute_values=impute_values,
        db_storage=db_storage,
        data_name=f"{data_name}_{only_confounders}_{'A' if additional_data else 'N'}",
    )
    comparisons_to_run = COMPARISONS
    if "altoida" in data_name:
        # Change the comparisons for Altoida due to missing MildAD data
        mild_ad_id = NEW_GROUPS_INVERSE["MildAD"] - 1
        comparisons_to_run = list(
            (a, b) for (a, b) in COMPARISONS if a != mild_ad_id and b != mild_ad_id
        )

    if repeat is not None:
        # Perform the NCV for a specific seed
        results, shap_values, test_shap_values = trainer.ncv(
            trials, comparisons_to_run, seed=repeat
        )
    else:
        # Run the repeated NCV for all seeds
        results, shap_values, test_shap_values = trainer.repeated_ncv(
            trials, repeats=repeats, comparisons=COMPARISONS
        )
    results["model"] = f"cart{'_base' if only_confounders else ''}"
    shap_values["model"] = f"cart{'_base' if only_confounders else ''}"
    test_shap_values["model"] = f"cart{'_base' if only_confounders else ''}"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(output_file)
    shap_values.to_csv(train_shap_file)
    test_shap_values.to_csv(test_shap_file)


@ncv_app.command()
def xgb(
    x_data: Path,
    y_data: Path,
    output_file: Path,
    train_shap_file: Path,
    test_shap_file: Path,
    folds: int = 10,
    cores: int = 4,
    trials: int = 50,
    repeats: int = 10,
    additional_data: Optional[Path] = None,
    only_confounders: bool = False,
    single_observation_data: str = "n",
    repeat: Optional[int] = None,
    impute_values: bool = False,
    db_storage: str = "sqlite:///optuna.db",
    data_name: str = "data",
):
    prepare_logging()
    config = dict()
    random.seed(repeat)
    np.random.seed(repeat)

    # Flag if only one obersvation should be available per participant
    reduce_df = True if single_observation_data == "y" else False

    # Prepare data for
    xdf, groups = prepare_data(
        x_data,
        y_data,
        only_confounders,
        single_observation_data=reduce_df,
        optional_data=additional_data,
        keepna=impute_values,
    )

    # Initialize the trainer instance
    trainer = XgbTrainer(
        xdf,
        groups,
        config,
        folds=folds,
        cores=cores,
        impute_values=impute_values,
        db_storage=db_storage,
        data_name=f"{data_name}_{only_confounders}_{'A' if additional_data else 'N'}",
    )
    comparisons_to_run = COMPARISONS
    if "altoida" in data_name:
        # Change the comparisons for Altoida due to missing MildAD data
        mild_ad_id = NEW_GROUPS_INVERSE["MildAD"] - 1
        comparisons_to_run = list(
            (a, b) for (a, b) in COMPARISONS if a != mild_ad_id and b != mild_ad_id
        )

    if repeat is not None:
        # Perform the NCV for a specific seed
        results, shap_values, test_shap_values = trainer.ncv(
            trials, comparisons_to_run, seed=repeat
        )
    else:
        # Run the repeated NCV for all seeds
        results, shap_values, test_shap_values = trainer.repeated_ncv(
            trials, repeats=repeats
        )
    results["model"] = f"xgb{'_base' if only_confounders else ''}"
    shap_values["model"] = f"xgb{'_base' if only_confounders else ''}"
    test_shap_values["model"] = f"xgb{'_base' if only_confounders else ''}"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(output_file)
    shap_values.to_csv(train_shap_file)
    test_shap_values.to_csv(test_shap_file)


@ncv_app.command()
def lr(
    x_data: Path,
    y_data: Path,
    output_file: Path,
    train_shap_file: Path,
    test_shap_file: Path,
    folds: int = 10,
    cores: int = 4,
    trials: int = 50,
    repeats: int = 10,
    additional_data: Optional[Path] = None,
    only_confounders: bool = False,
    single_observation_data: str = "n",
    repeat: Optional[int] = None,
    impute_values: bool = False,
    db_storage: str = "sqlite:///optuna.db",
    data_name: str = "data",
):
    prepare_logging()
    config = dict()
    random.seed(repeat)
    np.random.seed(repeat)

    # Flag if only one obersvation should be available per participant
    reduce_df = True if single_observation_data == "y" else False

    # Prepare data for
    xdf, groups = prepare_data(
        x_data,
        y_data,
        only_confounders,
        single_observation_data=reduce_df,
        optional_data=additional_data,
        keepna=impute_values,
    )
    trainer = LogRegTrainer(
        xdf,
        groups,
        config,
        folds=folds,
        cores=cores,
        impute_values=impute_values,
        db_storage=db_storage,
        data_name=f"{data_name}_{only_confounders}_{'A' if additional_data else 'N'}",
    )
    comparisons_to_run = COMPARISONS
    if "altoida" in data_name:
        # Change the comparisons for Altoida due to missing MildAD data
        mild_ad_id = NEW_GROUPS_INVERSE["MildAD"] - 1
        comparisons_to_run = list(
            (a, b) for (a, b) in COMPARISONS if a != mild_ad_id and b != mild_ad_id
        )

    if repeat is not None:
        # Perform the NCV for a specific seed
        results, shap_values, test_shap_values = trainer.ncv(
            trials, comparisons_to_run, seed=repeat
        )
    else:
        # Run the repeated NCV for all seeds
        results, shap_values, test_shap_values = trainer.repeated_ncv(
            trials, repeats=repeats, comparisons=COMPARISONS
        )
    results["model"] = f"lr{'_base' if only_confounders else ''}"
    shap_values["model"] = f"lr{'_base' if only_confounders else ''}"
    test_shap_values["model"] = f"lr{'_base' if only_confounders else ''}"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(output_file)
    shap_values.to_csv(train_shap_file)
    test_shap_values.to_csv(test_shap_file)


# Fit models


@fit_app.command()
def fit_rf(
    x_data: Path,
    y_data: Path,
    output_folder: Path,
    folds: int = 10,
    cores: int = 4,
    trials: int = 50,
    additional_data: Optional[Path] = None,
    only_confounders: bool = False,
    single_observation_data: str = "n",
    seed: int = 42,
    impute_values: bool = False,
    db_storage: str = "sqlite:///optuna.db",
    data_name: str = "data",
    feature_exclusion_file: Optional[Path] = None,
):
    prepare_logging()
    config = dict()
    random.seed(seed)
    np.random.seed(seed)

    # Flag if only one obersvation should be available per participant
    reduce_df = True if single_observation_data == "y" else False

    # Prepare data for
    xdf, groups = prepare_data(
        x_data,
        y_data,
        only_confounders,
        single_observation_data=reduce_df,
        optional_data=additional_data,
        keepna=impute_values,
    )
    trainer = RfTrainer(
        xdf,
        groups,
        config,
        folds=folds,
        cores=cores,
        impute_values=impute_values,
        db_storage=db_storage,
        data_name=f"{data_name}_{only_confounders}_{'A' if additional_data else 'N'}",
    )
    comparisons_to_run = COMPARISONS
    if "altoida" in data_name:
        mild_ad_id = NEW_GROUPS_INVERSE["MildAD"] - 1
        proad_mildad_id = NEW_GROUPS_INVERSE["ProAD+MildAD"] - 1
        comparisons_to_run = list(
            (a, b)
            for (a, b) in COMPARISONS
            if (a != mild_ad_id and b != mild_ad_id)
            or (a != proad_mildad_id and b != proad_mildad_id)
        )

    if feature_exclusion_file is not None:
        with open(feature_exclusion_file, "r") as f:
            excluded_features = f.read().splitlines()
    else:
        excluded_features = []

    models = trainer.fit_models(
        trials, comparisons_to_run, seed=seed, excluded_features=excluded_features
    )

    # Save trained models to file for later usage
    output_folder.mkdir(exist_ok=True, parents=True)
    for model, comparison in zip(models, comparisons_to_run):
        pretty_comparison = (
            NEW_GROUPS[comparison[0] + 1] + "_" + NEW_GROUPS[comparison[1] + 1]
        )
        model_file = output_folder / f"{pretty_comparison}.joblib"
        joblib.dump(model, model_file)


@fit_app.command()
def fit_cart(
    x_data: Path,
    y_data: Path,
    output_folder: Path,
    folds: int = 10,
    cores: int = 4,
    trials: int = 50,
    additional_data: Optional[Path] = None,
    only_confounders: bool = False,
    single_observation_data: str = "n",
    seed: int = 42,
    impute_values: bool = False,
    db_storage: str = "sqlite:///optuna.db",
    data_name: str = "data",
    feature_exclusion_file: Optional[Path] = None,
):
    prepare_logging()
    config = dict()
    random.seed(seed)
    np.random.seed(seed)

    # Flag if only one obersvation should be available per participant
    reduce_df = True if single_observation_data == "y" else False

    # Prepare data for
    xdf, groups = prepare_data(
        x_data,
        y_data,
        only_confounders,
        single_observation_data=reduce_df,
        optional_data=additional_data,
        keepna=impute_values,
    )
    trainer = CartTrainer(
        xdf,
        groups,
        config,
        folds=folds,
        cores=cores,
        impute_values=impute_values,
        db_storage=db_storage,
        data_name=f"{data_name}_{only_confounders}_{'A' if additional_data else 'N'}",
    )
    comparisons_to_run = COMPARISONS
    if "altoida" in data_name:
        # Change the comparisons for Altoida due to missing MildAD data
        mild_ad_id = NEW_GROUPS_INVERSE["MildAD"] - 1
        comparisons_to_run = list(
            (a, b) for (a, b) in COMPARISONS if a != mild_ad_id and b != mild_ad_id
        )

    # Perform the NCV for a specific seed
    if feature_exclusion_file is not None:
        with open(feature_exclusion_file, "r") as f:
            excluded_features = f.read().splitlines()
    else:
        # Run the repeated NCV for all seeds
        excluded_features = []

    models = trainer.fit_models(
        trials, comparisons_to_run, seed=seed, excluded_features=excluded_features
    )

    # Save trained models to file for later usage
    output_folder.mkdir(exist_ok=True, parents=True)
    for model, comparison in zip(models, comparisons_to_run):
        pretty_comparison = (
            NEW_GROUPS[comparison[0] + 1] + "_" + NEW_GROUPS[comparison[1] + 1]
        )
        model_file = output_folder / f"{pretty_comparison}.joblib"
        joblib.dump(model, model_file)


@fit_app.command()
def fit_xgb(
    x_data: Path,
    y_data: Path,
    output_folder: Path,
    folds: int = 10,
    cores: int = 4,
    trials: int = 50,
    additional_data: Optional[Path] = None,
    only_confounders: bool = False,
    single_observation_data: str = "n",
    seed: int = 42,
    impute_values: bool = False,
    db_storage: str = "sqlite:///optuna.db",
    data_name: str = "data",
    feature_exclusion_file: Optional[Path] = None,
):
    prepare_logging()
    config = dict()
    random.seed(seed)
    np.random.seed(seed)

    # Flag if only one obersvation should be available per participant
    reduce_df = True if single_observation_data == "y" else False

    # Prepare data for training
    xdf, groups = prepare_data(
        x_data,
        y_data,
        only_confounders,
        single_observation_data=reduce_df,
        optional_data=additional_data,
        keepna=impute_values,
    )

    # Initialize the trainer instance
    trainer = XgbTrainer(
        xdf,
        groups,
        config,
        folds=folds,
        cores=cores,
        impute_values=impute_values,
        db_storage=db_storage,
        data_name=f"{data_name}_{only_confounders}_{'A' if additional_data else 'N'}",
    )
    comparisons_to_run = COMPARISONS
    if "altoida" in data_name:
        # Change the comparisons for Altoida due to missing MildAD data
        mild_ad_id = NEW_GROUPS_INVERSE["MildAD"] - 1
        comparisons_to_run = list(
            (a, b) for (a, b) in COMPARISONS if a != mild_ad_id and b != mild_ad_id
        )

    # Perform the NCV for a specific seed
    if feature_exclusion_file is not None:
        with open(feature_exclusion_file, "r") as f:
            excluded_features = f.read().splitlines()
    else:
        # Run the repeated NCV for all seeds
        excluded_features = []

    models = trainer.fit_models(
        trials, comparisons_to_run, seed=seed, excluded_features=excluded_features
    )

    # Save trained models to file for later usage
    output_folder.mkdir(exist_ok=True, parents=True)
    for model, comparison in zip(models, comparisons_to_run):
        pretty_comparison = (
            NEW_GROUPS[comparison[0] + 1] + "_" + NEW_GROUPS[comparison[1] + 1]
        )
        model_file = output_folder / f"{pretty_comparison}.joblib"
        joblib.dump(model, model_file)


@fit_app.command()
def fit_lr(
    x_data: Path,
    y_data: Path,
    output_folder: Path,
    folds: int = 10,
    cores: int = 4,
    trials: int = 50,
    additional_data: Optional[Path] = None,
    only_confounders: bool = False,
    single_observation_data: str = "n",
    seed: int = 42,
    impute_values: bool = False,
    db_storage: str = "sqlite:///optuna.db",
    data_name: str = "data",
    feature_exclusion_file: Optional[Path] = None,
):
    prepare_logging()
    config = dict()
    random.seed(seed)
    np.random.seed(seed)

    # Flag if only one obersvation should be available per participant
    reduce_df = True if single_observation_data == "y" else False

    # Prepare data for
    xdf, groups = prepare_data(
        x_data,
        y_data,
        only_confounders,
        single_observation_data=reduce_df,
        optional_data=additional_data,
        keepna=impute_values,
    )
    trainer = LogRegTrainer(
        xdf,
        groups,
        config,
        folds=folds,
        cores=cores,
        impute_values=impute_values,
        db_storage=db_storage,
        data_name=f"{data_name}_{only_confounders}_{'A' if additional_data else 'N'}",
    )
    comparisons_to_run = COMPARISONS
    if "altoida" in data_name:
        # Change the comparisons for Altoida due to missing MildAD data
        mild_ad_id = NEW_GROUPS_INVERSE["MildAD"] - 1
        comparisons_to_run = list(
            (a, b) for (a, b) in COMPARISONS if a != mild_ad_id and b != mild_ad_id
        )

    # Perform the NCV for a specific seed
    if feature_exclusion_file is not None:
        with open(feature_exclusion_file, "r") as f:
            excluded_features = f.read().splitlines()
    else:
        # Run the repeated NCV for all seeds
        excluded_features = []

    models = trainer.fit_models(
        trials, comparisons_to_run, seed=seed, excluded_features=excluded_features
    )

    # Save trained models to file for later usage
    output_folder.mkdir(exist_ok=True, parents=True)
    for model, comparison in zip(models, comparisons_to_run):
        pretty_comparison = (
            NEW_GROUPS[comparison[0] + 1] + "_" + NEW_GROUPS[comparison[1] + 1]
        )
        model_file = output_folder / f"{pretty_comparison}.joblib"
        joblib.dump(model, model_file)


if __name__ == "__main__":
    app()
