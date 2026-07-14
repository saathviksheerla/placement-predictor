import streamlit as st
import pickle
import pandas as pd # Import pandas to fix the warning

@st.cache_resource
def load_assets():
    model = pickle.load(open("placement_predictor_ml_model.pkl", "rb"))
    scaler = pickle.load(open("placement_predictor_ml_scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_assets()

st.title("🎓 Student Placement Predictor")

st.info(
    """
    🔬 **Experimental Note & Findings:**
    
    This app is a proof-of-concept trained on a trial dataset using **Logistic Regression**. 
    During testing, we discovered that the dataset has a strong feature bias:
    
    * **CGPA** is the dominant factor (Weight: `~3.27`). A CGPA $\ge$ 6.1 almost entirely dictates a positive placement outcome.
    * **IQ** has an optimized weight near zero (`~-0.05`), meaning it currently holds virtually no mathematical influence on the predictions.
    
    *Conclusion:* The model operates essentially as a single-feature classifier due to the underlying dataset distribution.
    """
)

cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
iq = st.number_input("Enter IQ Score", min_value=0, max_value=200, value=100, step=1)

if st.button("Predict Placement Status"):
    # FIX: Convert inputs into a DataFrame with matching column names
    raw_features = pd.DataFrame([[cgpa, iq]], columns=['cgpa', 'iq'])
    
    # Scale features using the DataFrame (Warning goes away!)
    scaled_features = scaler.transform(raw_features)
    
    # Predict
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.success("🎉 Placed!")
    else:
        st.error("😔 Not Placed.")


st.warning(
    """
    ⚠️ **🧠 Quick Recall Note:**
    * **CGPA rules everything:** The model weight for CGPA is `3.27`. Anything $\ge$ 6.1 forces a "Placed" result.
    * **IQ is ignored:** The model weight for IQ is practically `0` (`-0.05`). Changing this number will not change the prediction.
    * **Why?** The training dataset has a perfect linear split based only on CGPA.
    * **Logistic Regression** is a linear model, and it cannot learn non-linear relationships.
    """
)