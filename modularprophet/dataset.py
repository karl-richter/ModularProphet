import logging
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

logger = logging.getLogger("experiments")


class TimeSeries(Dataset):
    def __init__(
        self,
        df,
        n_lags=2,
        n_forecasts=3,
        lagged_features=None,
        future_features=None,
        target="target",
    ):
        self.df = df.copy()
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.lagged_features = list(set(lagged_features))
        self.future_features = list(set(future_features))
        # Postprocess the features
        if "lags" in self.future_features:
            self.future_features.remove("lags")
        self.lagged_future_features = self.future_features.copy()
        self.future_features += ["target"]

        self.target = target

        # Transform the dataframe into a dict of numpy array, infer the correct dtype
        columns = list(
            set(
                self.lagged_features
                + self.future_features
                + self.lagged_future_features
            )
        )
        self.df_numpy = {
            col: self.infer_dtype(self.df[col].to_numpy()) for col in columns
        }

        # Identify the range of samples
        sample_range = range(self.n_lags, len(self) + self.n_lags)

        # Pre-compute the samples
        self.lagged_samples = [
            {
                feature: self.df_numpy[feature][idx - self.n_lags : idx]
                for feature in self.lagged_features
            }
            for idx in sample_range
        ]

        self.future_samples = [
            {
                feature: self.df_numpy[feature][idx : idx + self.n_forecasts]
                for feature in self.future_features
            }
            for idx in sample_range
        ]

        self.lagged_future_samples = [
            {
                feature: self.df_numpy[feature][idx - self.n_lags : idx]
                for feature in self.lagged_future_features
            }
            for idx in sample_range
        ]

    def __len__(self):
        # artificially shorten the length of the df account for lags and forecasts
        return len(self.df) - self.n_lags - self.n_forecasts + 1

    def __getitem__(self, idx):
        # Target
        y = self.future_samples[idx]  # TODO: check if needed [self.target]

        # Features
        # Delete the target to avoid leakage
        x = self.future_samples[idx].copy()
        del x["target"]
        x.update(self.lagged_samples[idx].copy())
        x["lagged"] = self.lagged_future_samples[idx].copy()

        return x, y

    def get_normalization_parameters(self):
        return self.shift, self.scale

    def infer_dtype(self, arr):
        first_sample = arr[0]
        if isinstance(first_sample, float):
            return np.array(arr, dtype=np.float32)
        elif isinstance(first_sample, int):
            return np.array(arr, dtype=np.int64)
        elif isinstance(first_sample, list):
            return np.array(np.stack(arr, axis=0), dtype=np.float32)
        else:
            raise ValueError(f"Unknown type {type(first_sample)}")


class TimeDataModule(pl.LightningDataModule):
    def __init__(self, df, config, n_forecasts, batch_size):
        super().__init__()
        self.config = config
        self.shift = None
        self.scale = None

        # Set parameters
        self.n_forecasts = n_forecasts
        self.n_lags = config.get("model.args.n_lags")
        self.batch_size = batch_size

        # Load data and pre-process
        self.df_raw = df

        self.df = self.df_raw.copy()
        self.df = self.df.rename(columns={"ds": "time", "y": "target"})
        self.df, self.shift, self.scale = self.pre_process_dataframe(self.df)

        self.lagged_features = ["lags"]
        self.future_features = ["time"]  # + [
        #     component.get("args.feature") for component in config.model.components
        # ]

        # Split data
        self.df_train = self.df[: -self.n_forecasts].copy()
        self.df_predict = self.df.copy()  # self.df[-(self.n_lags + self.n_forecasts) :]
        self.df_predict["target_raw"] = self.df_predict["target"].copy()

    def update_predict_df(self, df):
        self.df_predict = df.copy()
        self.df_predict = self.df_predict.rename(columns={"ds": "time", "y": "target"})
        self.df_predict = self.pre_process_dataframe(self.df_predict)[0]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            pass
        elif stage == "predict":
            pass

    def train_dataloader(self, n_lags=None, shuffle=True, target="target"):
        t0 = time.time()
        dataset_train = TimeSeries(
            self.df_train,
            n_lags=self.n_lags if n_lags is None else n_lags,
            n_forecasts=self.n_forecasts,
            lagged_features=self.lagged_features,
            future_features=self.future_features,
            target=target,
        )
        logger.info(
            f"Loading training dataset took {round(time.time() - t0, 3)} seconds"
        )
        return DataLoader(
            dataset_train,
            batch_size=self.batch_size
            if self.config.get("training.optimizer") != "bfgs"
            else len(dataset_train),
            shuffle=shuffle,
        )

    def predict_dataloader(self):
        t0 = time.time()
        dataset_predict = TimeSeries(
            self.df_predict,
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            lagged_features=self.lagged_features,
            future_features=self.future_features,
        )
        logger.info(
            f"Loading prediction dataset took {round(time.time() - t0, 3)} seconds"
        )
        return DataLoader(
            dataset_predict,
            batch_size=self.batch_size
            if self.config.get("training.optimizer") != "bfgs"
            else len(dataset_predict),
            shuffle=False,
        )

    def get_normalization_parameters(self):
        return {"shift": self.shift, "scale": self.scale}

    def denormalize(self, df, extra_cols=[]):
        """
        Denormalize the predictions.
        """
        df["Level"] = self.shift["target"]
        df["x"] = df["x"] * self.scale["time"] + self.shift["time"]
        df["y"] = df["y"] * self.scale["target"] + self.shift["target"]
        df["y_hat"] = df["y_hat"] * self.scale["target"] + self.shift["target"]
        columns_to_denormalize = [
            col
            for col in df.columns
            if any(
                [
                    col.split(" ")[0] in comp.get("name")
                    for comp in self.config.get("model.components")
                ]
            )
            and (col not in ["y", "y_hat", "x"])
        ]
        columns_to_denormalize += extra_cols
        for component in columns_to_denormalize:
            df[component] = df[component] * self.scale["target"] + self.shift["target"]
        return df

    def remove_prediction_from_input(self, prediction, cols=["lags"]):
        """
        Extract the predictions by the current component and remove from the training data
        for the next component to train.

        Note: Always uses the first prediction of the batch to remove from the target, except
        the last batch, where all predictions are used. This is required to match the shape of
        of the training data.
        """
        # Get the 1-step ahead prediction of each batch
        # Append the n-step ahead prediction of the last batch
        y_hat = torch.cat(
            [batch["y_hat"][:, 0] for batch in prediction]
            + [prediction[-1]["y_hat"][-1:, 1:][0]],
            axis=0,
        )
        # Since the lags have not been forecasted, repeat the first value of the target
        y_hat = torch.cat([y_hat[0].repeat(self.n_lags), y_hat])
        # Remove the learned component from the target in the training data
        for col in cols:
            self.df_train[col] = self.df_train[col] - pd.Series(y_hat.detach().numpy())
        # Repeat the last value of the target to match the length of the prediction data
        y_hat = torch.cat([y_hat, y_hat[-1].repeat(self.n_forecasts)])
        # Remove the learned component from the target in the prediction data
        for col in cols:
            self.df_predict[col] = self.df_predict[col] - pd.Series(
                y_hat.detach().numpy()
            )

    def pre_process_dataframe(self, df, normalization="standard"):
        logger.info(f"Pre-processing dataframe with {len(df)} samples")

        df = df.copy()

        # Convert str to datetime
        df["dt"] = pd.to_datetime(df["time"], origin="unix")
        df["time"] = df["dt"].astype(int) / 10**9
        df["day"] = df["dt"].dt.day
        df["month"] = df["dt"].dt.month
        df["weekofyear"] = df["dt"].dt.isocalendar().week
        df["dayofweek"] = df["dt"].dt.dayofweek
        df["hour"] = df["dt"].dt.hour

        # Fourier terms
        # TODO: df = self.add_fourier_features(df, config=self.config.model.components)

        # Set normalization parameters
        # Only normalize the columns that are used for normalization
        normalization_columns = ["time", "target"]
        norm_df = df[normalization_columns].copy()

        # Scaling
        if not self.shift or not self.scale:
            shift, scale = self.get_scaling_params(
                norm_df, normalization={"target": normalization, "time": "minmax"}
            )
        else:
            shift, scale = self.shift, self.scale

        # Normalize
        for feature in shift.keys():
            df[feature] = (df[feature] - shift[feature]) / scale[feature]

        # Copy the target into a lag column
        df["lags"] = df["target"].copy()

        ### Smoothing ###
        # smoothing_factor = max(int(len(df) / 2 * 0.05), 1) * 2

        # Hamming filter smoothing
        # padded_arget = np.pad(
        #     np.array(df["target"]), pad_width=round(smoothing_factor / 2), mode="edge"
        # )
        # window = np.hamming(smoothing_factor)
        # df["hamming"] = np.convolve(
        #     window / window.sum(),
        #     padded_arget,
        #     mode="valid",
        # )[1:]

        return df, shift, scale

    def get_scaling_params(self, norm_df, normalization):
        """
        https://en.wikipedia.org/wiki/Feature_scaling
        """
        # Only derive the scaling parameters from specific columns
        shift = {}
        scale = {}

        for feature in normalization.keys():
            if normalization[feature] == "minmax":
                shift[feature] = norm_df[feature].min()
                scale[feature] = norm_df[feature].max() - norm_df[feature].min()
            elif normalization[feature] == "standard":
                shift[feature] = norm_df[feature].mean()
                scale[feature] = norm_df[feature].std()
            elif normalization[feature] == "mean":
                shift[feature] = norm_df[feature].mean()
                scale[feature] = norm_df[feature].max() - norm_df[feature].min()
            else:
                raise ValueError(
                    f"Normalization {normalization[feature]} not supported"
                )

        return shift, scale

    def add_fourier_features(self, df, config):
        df = df.copy()
        for component in config:
            if component.get("name") == "FourierSeasonality":
                period = component.get("args.period")
                series_order = component.get("args.series_order")
                df[component.get("args.feature")] = df["time"].apply(
                    lambda t: [
                        fun((2.0 * (i + 1) * np.pi * (t / 3600 / 24) / period))
                        for i in range(series_order)
                        for fun in (np.sin, np.cos)
                    ]
                )
        return df

    def get_autocorrelation(self, target: str, df=None, lags=None):
        """
        Get the autocorrelation of the target variable.
        """
        if df is None:
            df = self.df_train
        if lags is None:
            lags = self.n_lags

        series = df[target].copy()
        autocorr = [series.autocorr(i) for i in range(1, lags + 1)]
        autocorr = np.flip(autocorr)
        return autocorr / np.sum(np.abs(autocorr))
