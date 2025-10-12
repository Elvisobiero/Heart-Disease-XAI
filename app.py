import os
import streamlit as st
import shap
import joblib
import pandas as pd
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dir = os.getcwd()
model = joblib.load(os.path.join(dir, 'best_xgb.pkl'))
explainer = joblib.load(os.path.join(dir, 'explainer2.pkl'))



def main():
    st.set_page_config(layout = 'wide')
    st.title("Heart Disease Prediction")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🔢 Input Features")

        age = st.number_input("Age", min_value=1, max_value=100, value=28)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("CP", [0, 1, 2, 3])
        oldpeak = st.number_input("Oldpeak", value=1.0)
        thalach = st.number_input("thalach", value=150)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("ca", [0, 1, 2, 3])
        thal = st.selectbox('thal', [1, 2, 3])
        exang = st.selectbox('exang', [0, 1])

        # prediction = model.predict([[age, sex, cp, thalach, exang, oldpeak, slope, ca, thal]])
        # st.header("Prediction is {}".format(prediction))

        input_data = pd.DataFrame([{
            'age': age, 
            'sex': sex, 
            'cp': cp, 
            'thalach': thalach, 
            'exang': exang, 
            'oldpeak': oldpeak, 
            'slope': slope, 
            'ca': ca, 
            'thal': thal
            }])


    with col2:
        st.header("📈 Model prediction & Explanation")
        pred_proba = model.predict_proba(input_data)[0][1] * 100
        pred_label = "✅ Heart Disease" if pred_proba >= 50 else "❌ No Heart Disease"
        st.metric(label="PREDICTION", value=pred_label, delta=f"{pred_proba:.2f}% likelihood of Heart Disease")

        # SHAP explanation
        shap_values = explainer(input_data)
        st.markdown("### 💡 Top Feature Contributions")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        # st.pyplot(bbox_inches='tight')
        st.pyplot(fig)
        # plt.clf()



if __name__ == '__main__':
    main()
