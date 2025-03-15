# Imports --------------------------------------------------------------------------------------------------------------
import streamlit as st
from datetime import datetime, timedelta
import joblib
import time
import numpy as np
import warnings
import pandas as pd
import gunicorn

# Config ---------------------------------------------------------------------------------------------------------------
warnings.simplefilter("ignore")

st.set_page_config(layout="wide", page_title="Air Pollution Predictor", page_icon=":tornado:",
                   menu_items={'About': "This app uses a LightGBM model trained on past air pollution data in Leeds "
                                        "city "
                                        "center, and weather data collected from Spen Farm, NE Leeds. The purpose is to"
                                        " give a short forecast on air quality given current time and weather conditions"
                                        " and whether it would still be appropriate to exercise in."})

MODEL_FP = "https://raw.githubusercontent.com/twrighta/leeds-air-pollution-streamlit/main/lgbm1.joblib"
SCALER_FP = "https://raw.githubusercontent.com/twrighta/leeds-air-pollution-streamlit/main/lgbm1_scaler.joblib"


# Functions ------------------------------------------------------------------------------------------------------------
@st.cache_resource
def load_trained(model_path, scaler_path):
    """
    :param model_path: path to saved model
    :param  scaler_path: path to trained scaler
    :return: trained model object, trained scaler object
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# Text write-streaming
def stream_heading(sentence):
    for word in sentence.split(" "):
        yield word + " "
        time.sleep(0.05)


# Calculate future time for output
def calc_future_time(time, hrs_ahead):
    if time + hrs_ahead < 24:
        if time + hrs_ahead <= 12:
            return str(time + hrs_ahead) + "AM:"
        else:
            return str(time + hrs_ahead) + "PM:"

    else:
        if (time + hrs_ahead) - 24 <= 12:
            return str((time + hrs_ahead) - 24) + "AM"
        else:
            return str(((time + hrs_ahead) - 24)) + "PM"


# tooltip explainer for the pollution score output - Max score 28 min is 7
def pollution_tooltip(score):
    if int(score) < 12:
        return "Good air quality, no issues for exercise outdoors"
    if 12 <= int(score) < 18:
        return "Ok air quality. Sensitive individuals may experience issues whilst exercising"
    if 18 <= int(score) < 24:
        return "Bad air quality. Many individuals may experience issues whilst exercising"
    else:
        return "Terrible air quality. Do not exercise outdoors."


# Generate Predictions from Input
@st.cache_data
def generate_predictions(_model, model_inputs):
    """
    :param _model: trained loaded model. Underscore tells streamlit not to hash this argument
    :param model_inputs: numpy array of input features
    :return: model predictions list - pollution score for each time interval
    """
    predictions = model.predict(model_inputs)  # inputs already scaled  # 5 number list
    rounded = np.round(predictions, 0)  # round all predictions to nearest whole number
    return rounded


@st.cache_data
def create_model_input(current_month, current_day, current_hour, air_temperature, absolute_humidity,
                       atmospheric_pressure, wind_speed, wind_direction, net_radiation, _scaler):
    """
    :param current_month: user input month (1-12)
    :param current_day: user input day
    :param current_hour: user input hour
    :param air_temperature: user input air temperature (C)
    :param absolute_humidity: user input absolute humidity
    :param atmospheric_pressure: user input air pressure Pa
    :param wind_speed: user input wind speed m/s
    :param wind_direction: user input wind direction (degrees)
    :param net_radiation: user input net radiation w/m2
    :param _scaler: trained scaler. Underscored so Streamlit does not try to hash it.
    :return: 1D Numpy Array that matches the model's expected input shape and value posititions
    """

    # Functions to process the reformatted date data and information on future dates.
    # Check if the datetime is a weekday
    def is_weekday(date: datetime) -> tuple[int, int]:
        if date.weekday() in [0, 1, 2, 3, 4]:
            return 1
        else:
            return 0

    # Return day of week
    def what_weekday(date: datetime) -> int:
        return date.day

    # return bool if Christmas period or not
    def is_christmas(date: datetime) -> int:
        if (date.month == 12) & (date.day >= 19):
            return 1
        elif (date.month == 1) & (date.day < 5):
            return 1
        else:
            return 0

    # return bool if rushour period or not
    def is_rushhour(date: datetime) -> int:
        if (date.weekday() in [0, 1, 2, 3, 4]) & (date.hour in [6, 7, 8, 9, 16, 17, 18, 19]):
            return 1
        else:
            return 0

    # Calculate and reformat current date:
    current_date = datetime.strptime(f"{datetime.now().year}-{current_month}-{current_day} {current_hour}:00",
                                     "%Y-%m-%d %H:00")  # YYYY-M-D hh:mm

    # Future dates for predictions
    date_plus_1 = current_date + timedelta(hours=1)
    date_plus_2 = current_date + timedelta(hours=2)
    date_plus_6 = current_date + timedelta(hours=6)
    date_plus_12 = current_date + timedelta(hours=12)
    date_plus_24 = current_date + timedelta(hours=24)
    dates = {"plus_0": current_date,
             "plus_1": date_plus_1,
             "plus_2": date_plus_2,
             "plus_6": date_plus_6,
             "plus_12": date_plus_12,
             "plus_24": date_plus_24}

    # Iterate through each date combination in dates, and calculate the column values using the above functions
    model_inputs = [current_month, current_day, current_hour, air_temperature, absolute_humidity, atmospheric_pressure,
                    wind_speed, wind_direction, net_radiation]
    for time_desc, time in dates.items():
        model_inputs.append(is_weekday(time))
        model_inputs.append(what_weekday(time))
        model_inputs.append(is_christmas(time))
        model_inputs.append(is_rushhour(time))

    # Now add the remaining interaction columns onto the array
    model_inputs.append((air_temperature * atmospheric_pressure))
    model_inputs.append((wind_direction * wind_speed))
    model_inputs.append((air_temperature * net_radiation))
    model_inputs.append((absolute_humidity * atmospheric_pressure))

    # Reshape to 1D
    model_inputs_reshaped = np.array(model_inputs).reshape(1, -1)

    # Then scale whole input by the pretrained scaler
    scaler.transform(model_inputs_reshaped)  # originally model_inputs

    # Return model input
    return model_inputs_reshaped


# Create app layout with functionality ---------------------------------------------------------------------------------

# Load Model and Scaler into app
model, scaler = load_trained(MODEL_FP, SCALER_FP)

# Webpage title
st.title("Air Pollution Forecasting: Leeds")
st.divider()


# Ask for Date/time User Inputs ----------------------------------------------------------------------------------------
st.write_stream(stream_heading("Please enter the current date and time"))
month_col, day_col, hour_col = st.columns(spec=3,
                                          gap="medium",
                                          vertical_alignment="center",
                                          border=True)
# month input
with month_col:
    month_input = st.number_input(label="Enter Month (1-12)",
                                  min_value=1,
                                  max_value=12,
                                  step=1)
# day input
with day_col:
    day_input = st.number_input(label="Enter Day (1-31)",
                                min_value=1,
                                max_value=31,
                                step=1)
# hour input
with hour_col:
    hour_input = st.number_input(label="Enter Hour (1-24)",
                                 min_value=1,
                                 max_value=24,
                                 step=1)
st.divider()

# Ask for weather user inputs
st.write_stream(stream_heading("Please enter your current weather conditions:"))

temp_col, humidity_col, pressure_col = st.columns(spec=3,
                                                  gap="small",
                                                  vertical_alignment="center",
                                                  border=True)

with temp_col:
    temp_input = st.slider(label="Enter Temperature (C)",
                           min_value=-10,
                           max_value=40,
                           value=10,
                           step=1)

with humidity_col:
    humidity_input = st.number_input(label="Enter Humidity",
                                     min_value=0.0,
                                     max_value=20.0,
                                     step=0.05,
                                     value=5.0)

with pressure_col:
    pressure_input = st.number_input(label="Enter Pressure (Pa)",
                                     min_value=950,
                                     max_value=1040,
                                     step=1,
                                     value=1000)

speed_col, direction_col, radiation_col = st.columns(spec=3,
                                                     gap="small",
                                                     vertical_alignment="center",
                                                     border=True)

with speed_col:
    speed_input = st.number_input(label="Enter Wind Speed (m/s)",
                                  min_value=0.0,
                                  max_value=20.0,
                                  step=0.1,
                                  value=1.0)

with direction_col:
    direction_input = st.number_input(label="Enter Wind Direction (Degrees)",
                                      min_value=0.0,
                                      max_value=360.0,
                                      step=0.5,
                                      value=180.0)

with radiation_col:
    radiation_input = st.slider(label="Enter Net Radiation (W/m2)",
                                min_value=-200.0,
                                max_value=800.0,
                                step=0.5,
                                value=100.0)

# ----------------------------------------------------------------------------------------------------------------------
# Create model input:
model_input = create_model_input(month_input, day_input, hour_input, temp_input, humidity_input, pressure_input,
                                 speed_input,
                                 direction_input, radiation_input, scaler)

# ----------------------------------------------------------------------------------------------------------------------
# Generate Predictions
generate_pred_btn = st.button(label="Generate Predictions",
                              help="Click to generate pollution scores for 1, 2, 6, 12 and 24 hours time.",
                              type="primary",
                              icon=":material/output_circle:")

# Initializing a session state to store predictions in. First have to initialize it
if "predictions" not in st.session_state:
    st.session_state.predictions = None  # Initialize session state
    print(f"Initial streamlit session state: {st.session_state.predictions}")

if generate_pred_btn:
    with st.spinner("Generating predictions", show_time=True):
        time.sleep(1)
    st.session_state["predictions"] = generate_predictions(model,
                                                           model_input)  # Store predictions in session   # old: all_preds = generate_predictions(model, model_input)
    predictions = st.session_state["predictions"].reshape(1, -1)  # Make 1D

    preds_df = pd.DataFrame(columns=["1hr Forecast", "2hr Forecast", "6hr Forecast", "12hr Forecast", "24hr Forecast"],
                            data=predictions)

    st.divider()
    # Subheading for displaying scores
    st.subheader("Pollution Scores:")

    # Columns for displaying scores in
    hr1, hr2, hr6, hr12, hr24 = st.columns(5)

    with hr1:
        st.write(f"At {calc_future_time(hour_input, 1)}")
        st.text(str(int(preds_df["1hr Forecast"].values)), help=pollution_tooltip(int(preds_df["1hr Forecast"].values)))
    with hr2:
        st.write(f"At {calc_future_time(hour_input, 2)}")
        st.text(str(int(preds_df["2hr Forecast"].values)), help=pollution_tooltip(int(preds_df["2hr Forecast"].values)))
    with hr6:
        st.write(f"At {calc_future_time(hour_input, 6)}")
        st.text(str(int(preds_df["6hr Forecast"].values)), help=pollution_tooltip(int(preds_df["6hr Forecast"].values)))
    with hr12:
        st.write(f"At {calc_future_time(hour_input, 12)}")
        st.text(str(int(preds_df["12hr Forecast"].values)),
                help=pollution_tooltip(int(preds_df["12hr Forecast"].values)))
    with hr24:
        st.write(f"At {calc_future_time(hour_input, 24)}")
        st.text(str(int(preds_df["24hr Forecast"].values)),
                help=pollution_tooltip(int(preds_df["24hr Forecast"].values)))


