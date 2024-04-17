import streamlit as st
import numpy as np

# Machine learning
import joblib
import os

attribute_info = """
                - Department: Sales & Marketing, Operations, Technology, Analytics, R&D, Procurement, Finance, Legal.
                - Region: Region 1 - Region 10.
                - Education: Below Secondary, Bachelor's, Master's or above.
                - Gender: Male or Female.
                - Recruitment Channel: Referred, Sourcing, Others.
                - No of Training: 1-10.
                - Age: 10-60.
                - Previous Year Rating: 1-5
                - Length of Service: 1-37 Month.
                - Awards Won: 1.Yes, 0.No.
                - Average Training Score: 0-100
                """


dep = {
    "Sales & Marketing": 1,
    "Operations": 2,
    "Technology": 3,
    "Analytics": 4,
    "R&D": 5,
    "Procurement": 6,
    "Finance": 7,
    "HR": 8,
    "Legal": 9,
}
edu = {"Below Secondary": 1, "Bachelors": 2, "Masters or Above": 3}
rec = {"Referred": 1, "Sourcing": 2, "Other": 3}
gen = {"M": 1, "F": 2}
reg = {
    "region_1": 1,
    "region_2": 2,
    "region_3": 3,
    "region_4": 4,
    "region_5": 5,
    "region_6": 6,
    "region_7": 7,
    "region_8": 8,
    "region_9": 9,
    "region_10": 10,
    "region_11": 11,
    "region_12": 12,
    "region_13": 13,
    "region_14": 14,
    "region_15": 15,
    "region_16": 16,
    "region_17": 17,
    "region_18": 18,
    "region_19": 19,
    "region_20": 20,
}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def run_ml_app():
    st.subheader("ML Section")

    with st.expander("Attribut Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    department = st.selectbox(
        "Department",
        [
            "Sales & Marketing",
            "Operations",
            "Technology",
            "Analytics",
            "R&D",
            "Procurement",
            "Finance",
            "Legal",
        ],
    )
    region = st.selectbox("Region", reg.keys())
    education = st.selectbox("Education", edu.keys())
    gender = st.radio("Gender", gen.keys())
    recruitment = st.selectbox("Recruitment Channel", rec.keys())
    training = st.number_input("No of Training", 1, 10)
    age = st.number_input("Age", 10, 60)
    rating = st.number_input("Previous Year Rating", 1, 5)
    service = st.number_input("Length of Service", 1, 37)
    awards = st.radio("Awards Won", [0, 1])
    avg_training = st.number_input("Average Training Score", 0, 100)

    with st.expander("Your Selected Option"):
        result = {
            "Department": department,
            "Region": region,
            "Educational": education,
            "Gender": gender,
            "Recruitment": recruitment,
            "Training": training,
            "Age": age,
            "Rating": rating,
            "Service": service,
            "Award": awards,
            "Avg. Training": avg_training,
        }
    # st.write(result)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in [
            "Sales & Marketing",
            "Operations",
            "Technology",
            "Analytics",
            "R&D",
            "Procurement",
            "Finance",
            "Legal",
        ]:
            res = get_value(i, dep)
            encoded_result.append(res)
        elif i in [
            "region_1",
            "region_2",
            "region_3",
            "region_4",
            "region_5",
            "region_6",
            "region_7",
            "region_8",
            "region_9",
            "region_10",
            "region_11",
            "region_12",
            "region_13",
            "region_14",
            "region_15",
            "region_16",
            "region_17",
            "region_18",
            "region_19",
            "region_20",
        ]:
            res = get_value(i, reg)
            encoded_result.append(res)
        elif i in ["Below Secondary", "Bachelors", "Master or above"]:
            res = get_value(i, edu)
            encoded_result.append(res)
        elif i in ["M", "F"]:
            res = get_value(i, gen)
            encoded_result.append(res)
        elif i in ["Referred", "Sourcing", "Other"]:
            res = get_value(i, rec)
            encoded_result.append(res)

    # st.write(encoded_result)

    # st.subheader("Prediction Result")
    single_sample = np.array(encoded_result).reshape(1, -1)
    # st.write(single_sample)

    model = load_model("model_grad.pkl")

    prediction = model.predict(single_sample)
    pred_prob = model.predict_proba(single_sample)

    # st.subheader("Prediction Absolute")
    # st.write(prediction)
    st.subheader("Prediction Probability")
    # st.write(pred_prob)

    pred_probability = {
        "Promoted": round(pred_prob[0][1] * 100, 4),
        "Not Promoted": round(pred_prob[0][0] * 100, 4),
    }

    if prediction == 1:
        st.success("Congratulations, You are Promoted")
        st.write(pred_probability)
        st.balloons()
    else:
        st.warning("Please More Contribute to the Company", icon="⚠️")
        st.write(pred_probability)
