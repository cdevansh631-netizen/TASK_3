import streamlit as st
import pandas as pd
import joblib

# load saved model and preprocessing objects
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# same columns and order as training
num_cols = ["innings", "over", "balls_bowled", "ball_left", "current_score",
            "wickets_fallen", "current_run_rate", "runs_off_bat", "extras"]
cat_cols = ["batting_team", "bowling_team", "venue_x"]

st.title("IPL Final Score Predictor")
st.write("Enter the match situation and the model will predict the final total of the innings.")

# inputs
innings = st.selectbox("Innings", [1, 2])
batting_team = st.selectbox("Batting Team", encoders["batting_team"].classes_)
bowling_team = st.selectbox("Bowling Team", encoders["bowling_team"].classes_)
venue = st.selectbox("Venue", encoders["venue_x"].classes_)
current_score = st.number_input("Current Score", min_value=0, max_value=300, value=80)
balls_bowled = st.slider("Balls Bowled (legal balls)", 1, 120, 60)
wickets_fallen = st.slider("Wickets Fallen", 0, 10, 2)

if st.button("Predict"):
    if batting_team == bowling_team:
        st.warning("Batting team and bowling team cannot be the same.")
    else:
        over = (balls_bowled - 1) // 6 + 1
        ball_left = 120 - balls_bowled
        run_rate = current_score * 6 / balls_bowled

        row = {
            "innings": innings,
            "over": over,
            "balls_bowled": balls_bowled,
            "ball_left": ball_left,
            "current_score": current_score,
            "wickets_fallen": wickets_fallen,
            "current_run_rate": run_rate,
            "runs_off_bat": 0,
            "extras": 0,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue_x": venue,
        }
        data = pd.DataFrame([row])[num_cols + cat_cols]

        # same steps as training: encode, then scale
        for col in cat_cols:
            data[col] = encoders[col].transform(data[col])
        data[num_cols] = scaler.transform(data[num_cols])

        prediction = model.predict(data)[0]
        st.success("Predicted Final Total : " + str(round(prediction)) + " runs")