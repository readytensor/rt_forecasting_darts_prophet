import os
import warnings
import joblib
import numpy as np
import pandas as pd
from pandas import RangeIndex, DatetimeIndex
from typing import Optional, List, Union, Callable, Sequence
from darts.models.forecasting.prophet_model import Prophet
from darts import TimeSeries
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the Prophet Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "Prophet Forecaster"
    made_up_frequency = "S"  # by seconds
    made_up_start_dt = "2000-01-01 00:00:00"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        add_seasonalities: Union[dict, List[dict], None] = None,
        country_holidays: Optional[str] = None,
        suppress_stdout_stderror: Optional[bool] = True,
        add_encoders: Optional[dict] = None,
        cap: Union[
            float, Callable[[Union[DatetimeIndex, RangeIndex]], Sequence[float]], None
        ] = None,
        floor: Union[
            float, Callable[[Union[DatetimeIndex, RangeIndex]], Sequence[float]], None
        ] = None,
        **prophet_kwargs,
    ):
        """Construct a new Prophet Forecaster

        Args:
            data_schema (ForecastingSchema): Schema of the data used for training.
            add_seasonalities (Union[dict, List[dict], None]):
            Optionally, a dict or list of dicts with custom seasonality/ies to add to the model. Each dict takes the following mandatory and optional data:
            dict({
                'name': str  # (name of the seasonality component),
                'seasonal_periods': Union[int, float]  # (nr of steps composing a season),
                'fourier_order': int  # (number of Fourier components to use),
                'prior_scale': Optional[float]  # (a prior scale for this component),
                'mode': Optional[str]  # ('additive' or 'multiplicative')
            })
            An example for seasonal_periods: If you have hourly data (frequency='H') and your seasonal cycle repeats after 48 hours then set seasonal_periods=48.
            Notice that this value will be multiplied by the inferred number of days for the TimeSeries frequency (1 / 24 in this example) to be consistent with the add_seasonality() method of Facebook Prophet,
            where the period parameter is specified in days.
            Apart from seasonal_periods, this is very similar to how you would call Facebook Prophet's add_seasonality() method.
            Alternatively, you can add seasonalities after model creation and before fitting with add_seasonality().


            country_holidays (Optional[str]): An optional country code, for which holidays can be taken into account by Prophet.
            See: https://github.com/dr-prodigy/python-holidays
            In addition to those countries, Prophet includes holidays for these countries: Brazil (BR), Indonesia (ID), India (IN),
            Malaysia (MY), Vietnam (VN), Thailand (TH), Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD),
            Egypt (EG), China (CN), and Russia (RU).

            suppress_stdout_stderror (bool): Optionally suppress the log output produced by Prophet during training.

            cap (Union[float, Callable[[Union[DatetimeIndex, RangeIndex]], Sequence[float]], None]):
            Parameter specifiying the maximum carrying capacity when predicting with logistic growth. Mandatory when growth = 'logistic', otherwise ignored. See <https://facebook.github.io/prophet/docs/saturating_forecasts.html> for more information on logistic forecasts.
            Can be either
            - a number, for constant carrying capacities
            - a function taking a DatetimeIndex or RangeIndex and returning a corresponding a Sequence of numbers,
            where each number indicates the carrying capacity at this index.

            floor (Union[float, Callable[[Union[DatetimeIndex, RangeIndex]], Sequence[float]], None]):
            Parameter specifiying the minimum carrying capacity when predicting logistic growth. Optional when growth = 'logistic' (defaults to 0), otherwise ignored. See <https://facebook.github.io/prophet/docs/saturating_forecasts.html> for more information on logistic forecasts.
            Can be either
            - a number, for constant carrying capacities
            - a function taking a DatetimeIndex or RangeIndex and returning a corresponding a Sequence of numbers,
            where each number indicates the carrying capacity at this index.

            add_encoders (Optional[dict]): A large number of future covariates can be automatically generated with add_encoders.
                This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that will be used as index encoders.
                Additionally, a transformer such as Darts' Scaler can be added to transform the generated covariates.
                This happens all under one hood and only needs to be specified at model creation. Read SequentialEncoder to find out more about add_encoders.
                Default: None. An example showing some of add_encoders features:

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'future': ['relative']},
                    'custom': {'future': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }

                prophet_kwargs: Some optional keyword arguments for Prophet. For information about the parameters see: The Prophet source code.
        """
        self.add_seasonalities = add_seasonalities
        self.country_holidays = country_holidays
        self.suppress_stdout_stderror = suppress_stdout_stderror
        self.add_encoders = add_encoders
        self.cap = cap
        self.floor = floor
        self.prophet_kwargs = prophet_kwargs
        self._is_trained = False
        self.models = {}
        self.time_to_int_map = {}
        self.data_schema = data_schema

    def prepare_data(self, data: pd.DataFrame, is_train=True) -> pd.DataFrame:
        """
        Function to prepare the dataframe to use with Prophet.

        If the time column is of type int, we will update it to be datetime
        by creating artificial dates starting at '1/1/2023 00:00:00'
        that increment by 1 second for each row.

        Additionally, there may be 0, 1 or more future covariates as were originally
        passed. These are returned as-is.
        """
        # sort data
        data = data.sort_values(by=[self.data_schema.id_col, self.data_schema.time_col])

        if self.data_schema.time_col_dtype == "INT":
            # Find the number of rows for each location (assuming all locations have
            # the same number of rows)
            series_val_counts = data[self.data_schema.id_col].value_counts()
            series_len = series_val_counts.iloc[0]
            num_series = series_val_counts.shape[0]

            if is_train:
                # since prophet requires a datetime column, we will make up a timeline
                start_date = pd.Timestamp(self.made_up_start_dt)
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
                self.last_timestamp = datetimes[-1]
                self.timedelta = datetimes[-1] - datetimes[-2]
            else:
                start_date = self.last_timestamp + self.timedelta
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
            int_vals = sorted(data[self.data_schema.time_col].unique().tolist())
            self.time_to_int_map = dict(zip(datetimes, int_vals))
            # Repeat the datetime range for each location
            data[self.data_schema.time_col] = list(datetimes) * num_series
        else:
            data[self.data_schema.time_col] = pd.to_datetime(
                data[self.data_schema.time_col]
            )
            data[self.data_schema.time_col] = data[
                self.data_schema.time_col
            ].dt.tz_localize(None)

        return data

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by Prophet.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """
        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency in ["secondly", "other"]:
            return "S"

    def fit(self, history: pd.DataFrame, data_schema: ForecastingSchema) -> None:
        """Fit the Forecaster to the training data.
        A separate Prophet model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.
        """
        np.random.seed(0)
        history = self.prepare_data(history.copy())
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.models = {}

        for id, series in zip(all_ids, all_series):
            model = self._fit_on_series(history=series, data_schema=data_schema)
            self.models[id] = model

        self.all_ids = all_ids
        self._is_trained = True
        self.data_schema = data_schema

    def _fit_on_series(self, history: pd.DataFrame, data_schema: ForecastingSchema):
        """Fit Prophet model to given individual series of data"""
        model = Prophet(
            add_seasonalities=self.add_seasonalities,
            country_holidays=self.country_holidays,
            suppress_stdout_stderror=self.suppress_stdout_stderror,
            cap=self.cap,
            floor=self.floor,
            add_encoders=self.add_encoders,
            **self.prophet_kwargs,
        )

        series = TimeSeries.from_dataframe(
            history,
            data_schema.time_col,
            data_schema.target,
            fill_missing_dates=True,
        )

        future_covariates = None
        if data_schema.future_covariates + data_schema.static_covariates:
            future_covariates = TimeSeries.from_dataframe(
                history,
                data_schema.time_col,
                data_schema.future_covariates + data_schema.static_covariates,
            )
        model.fit(series, future_covariates=future_covariates)

        return model

    def predict(self, test_data: pd.DataFrame, prediction_col_name: str) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        future_df = self.prepare_data(test_data.copy(), is_train=False)

        groups_by_ids = future_df.groupby(self.data_schema.id_col)
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=self.data_schema.id_col)
            for id_ in self.all_ids
        ]
        # forecast one series at a time
        all_forecasts = []
        for id_, series_df in zip(self.all_ids, all_series):
            forecast = self._predict_on_series(key_and_future_df=(id_, series_df))
            forecast.insert(0, self.data_schema.id_col, id_)
            all_forecasts.append(forecast)

        # concatenate all series' forecasts into a single dataframe
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)

        all_forecasts.rename(
            columns={self.data_schema.target: prediction_col_name}, inplace=True
        )

        if self.data_schema.time_col_dtype == "INT":
            all_forecasts[self.data_schema.time_col] = all_forecasts[
                self.data_schema.time_col
            ].map(self.time_to_int_map)

        return all_forecasts

    def _predict_on_series(self, key_and_future_df):
        """Make forecast on given individual series of data"""
        key, future_df = key_and_future_df

        future_covariates = None
        if self.data_schema.future_covariates + self.data_schema.static_covariates:
            future_covariates = TimeSeries.from_dataframe(
                future_df,
                self.data_schema.time_col,
                self.data_schema.future_covariates + self.data_schema.static_covariates,
            )

        if self.models.get(key) is not None:
            forecast = self.models[key].predict(
                len(future_df), future_covariates=future_covariates
            )
            forecast_df = forecast.pd_dataframe()
            forecast = forecast_df[self.data_schema.target]
            future_df[self.data_schema.target] = forecast.values

        else:
            # no model found - key wasnt found in history, so cant forecast for it.
            future_df = None

        return future_df

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(history=history, data_schema=data_schema)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)
