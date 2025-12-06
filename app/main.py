import os
import pickle
import re
import tempfile
from io import BytesIO

import arabic_reshaper
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from bidi.algorithm import get_display
from fpdf import FPDF
from PIL import Image

# ------------------- Paths -------------------
current_dir = os.path.dirname(__file__)

# Models
model_path = os.path.join(current_dir, "..", "models", "rf_diabetes_model.pkl")
scaler_path = os.path.join(current_dir, "..", "models", "rf_scaler.pkl")
columns_path = os.path.join(current_dir, "..", "models", "rf_columns.pkl")

# Images
logo_path = os.path.join(current_dir, "..", "image", "logo.png")
logo_path1 = os.path.join(current_dir, "..", "image", "logo1.png")

# Fonts
font_path = os.path.join(current_dir, "Fonts", "DejaVuSans.ttf")

# ------------------- Load Models -------------------
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(columns_path, "rb") as f:
    model_columns = pickle.load(f)

# ------------------- Session State -------------------
if "saved_advice" not in st.session_state:
    st.session_state.saved_advice = ""

# ------------------- PDF Functions -------------------
def generate_pdf(patient_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.add_font("DejaVu", "B", font_path, uni=True)

    # Header
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Smart Diabetes Risk Assessment System", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)

    # Logo
    if os.path.exists(logo_path1):
        logo_width = 60
        page_width = pdf.w - 2 * pdf.l_margin
        x_center = (page_width - logo_width) / 2 + pdf.l_margin
        pdf.image(logo_path1, x=x_center, y=25, w=logo_width)
        pdf.ln(50)

    # Main Title
    pdf.set_font("DejaVu", "B", 18)
    pdf.cell(0, 10, "Patient Medical Report", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_line_width(0.5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(10)

    # Basic Patient Info
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Basic Patient Info", 0, 1, 'L')
    pdf.ln(3)
    pdf.set_font("DejaVu", "", 12)
    basic_data = {
        "Name": patient_info.get("Name", ""),
        "Age": patient_info.get("Age", ""),
        "Gender": patient_info.get("Gender", "")
    }
    for key, value in basic_data.items():
        pdf.cell(60, 8, str(key), 1, 0, 'L')
        pdf.cell(0, 8, str(value), 1, 1, 'L')
    pdf.ln(10)

    # Clinical Data
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Clinical Data", 0, 1, 'L')
    pdf.ln(3)
    pdf.set_font("DejaVu", "", 12)
    clinical_data = {
        "Weight (kg)": patient_info.get("Weight", ""),
        "Height (cm)": patient_info.get("Height", ""),
        "BMI": patient_info.get("BMI", ""),
        "Glucose (mg/dL)": patient_info.get("Glucose", "")
    }
    for key, value in clinical_data.items():
        pdf.cell(60, 8, str(key), 1, 0, 'L')
        pdf.cell(0, 8, str(value), 1, 1, 'L')
    pdf.ln(10)

    # Risk Level
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Risk Level", 0, 1, 'L')
    pdf.ln(3)
    risk_level = patient_info.get("Risk Level", "")
    if "Low" in risk_level:
        pdf.set_text_color(0, 128, 0)
    elif "Medium" in risk_level:
        pdf.set_text_color(255, 140, 0)
    else:
        pdf.set_text_color(220, 20, 60)
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, risk_level, 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Medical Recommendations
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Medical Recommendations", 0, 1, 'L')
    pdf.ln(5)
    pdf.set_font("DejaVu", "", 11)
    advice = patient_info.get("Advice", "")
    pdf.multi_cell(0, 8, advice)
    pdf.ln(5)

    # Footer
    pdf.set_y(-25)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "This report supports medical decisions and does not replace consultation with a physician.", 0, 0, 'C')

    pdf_bytes = pdf.output(dest='S')
    return bytes(pdf_bytes)
 
   


# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title="النظام الذكي لتقييم خطر الإصابة بالسكري | Smart Diabetes Risk Assessment",
    layout="wide",
    page_icon="💉"
)

st.sidebar.title("لوحة التحكم | Control Panel")
page = st.sidebar.radio(
    "اختر الصفحة | Select Page", [
        "الصفحة الرئيسية | Home",
        "تقييم خطر المريض | Patient Risk Assessment",
        "التقرير الطبي للمريض | Medical Report",
        "التحليل الجماعي | Batch Analysis",
        "تحليل صورة الفحص الطبي | Medical Image Analysis"
    ]
)

# =================== HOME PAGE ===================
if page == "الصفحة الرئيسية | Home":
    if os.path.exists(logo_path):
        st.image(logo_path, width=350)

    st.title(
        "💉 النظام الذكي لتقييم خطر الإصابة بمرض السكري | Smart Diabetes Risk Assessment System"
    )

    st.markdown("""
    مرحبًا بكم في النظام الذكي لتقييم خطر الإصابة بمرض السكري.  
    Welcome to the Smart Diabetes Risk Assessment System.

    يعتمد هذا النظام على الذكاء الاصطناعي والتعلّم الآلي لتحليل البيانات الصحية وتقديم
    تقييم دقيق لمستوى الخطورة مع توصيات طبية داعمة لاتخاذ القرار.

    This system uses AI & Machine Learning to analyze patient data
    and generate accurate risk predictions with medical recommendations.

    🔹 التنبؤ بالحالة الصحية | Health Risk Prediction  
    🔹 تقارير طبية بصيغة PDF | PDF Medical Reports  
    🔹 تحليل جماعي | Batch Analysis  
    🔹 تحليل صور الفحوصات | Medical Image OCR Analysis  

    🔴 هذا النظام داعم للقرار الطبي ولا يغني عن استشارة الطبيب  
    🔴 This system does not replace professional medical consultation
    """)

# =================== PAGE 2: PATIENT RISK ASSESSMENT ===================
elif page == "تقييم خطر المريض | Patient Risk Assessment":
    st.header("🧑‍⚕️ إدخل بيانات المريض | Enter Patient Data")

    with st.form("patient_form"):
        st.session_state.home_name = st.text_input(
            "اسم المريض | Patient Name", st.session_state.get("home_name", "")
        )
        st.session_state.home_age = st.number_input(
            "العمر | Age", 1, 120, st.session_state.get("home_age", 35)
        )
        st.session_state.home_gender = st.selectbox(
            "الجنس | Gender", ["ذكر | Male", "أنثى | Female"],
            index=0 if st.session_state.get(
                "home_gender", "ذكر | Male") == "ذكر | Male" else 1
        )
        st.session_state.home_weight = st.number_input(
            "الوزن (كغ) | Weight (kg)", 1, 300, st.session_state.get(
                "home_weight", 70)
        )
        st.session_state.home_height = st.number_input(
            "الطول (سم) | Height (cm)", 50, 250, st.session_state.get(
                "home_height", 170)
        )
        st.session_state.home_glucose = st.number_input(
            "مستوى الجلوكوز | Glucose (mg/dL)", 50, 400, st.session_state.get(
                "home_glucose", 110)
        )
        st.session_state.home_hypertension = st.selectbox(
            "ارتفاع ضغط الدم | Hypertension", ["لا | No", "نعم | Yes"],
            index=0 if st.session_state.get(
                "home_hypertension", "لا | No") == "لا | No" else 1
        )
        st.session_state.home_family_diabetes = st.selectbox(
            "تاريخ عائلي للسكري | Family Diabetes", ["لا | No", "نعم | Yes"],
            index=0 if st.session_state.get(
                "home_family_diabetes", "لا | No") == "لا | No" else 1
        )

        save_btn = st.form_submit_button("💾 حفظ البيانات | Save Data")

    if save_btn:
        st.success("✅ تم حفظ البيانات! سيتم استخدامها في التقرير الطبي لاحقًا.")

# =================== PAGE 3: MEDICAL REPORT ===================
elif page == "التقرير الطبي للمريض | Medical Report":
    st.header("📄 Patient Medical Report")

    # =================== FORM ===================
    with st.form("patient_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 120, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 1, 300, 70)
        height = st.number_input("Height (cm)", 50, 250, 170)
        glucose = st.number_input("Glucose (mg/dL)", 50, 400, 110)
        hypertensive = st.selectbox("Hypertension", ["No", "Yes"])
        family_diabetes = st.selectbox("Family Diabetes", ["No", "Yes"])

        submit = st.form_submit_button("💾 Save & Generate PDF")

    # =================== AFTER FORM SUBMISSION ===================
    if submit:
        # Calculate BMI
        bmi = round(weight / ((height / 100) ** 2), 2)

        # Prepare data for model
        new_data = pd.DataFrame({
            'age': [age],
            'gender': [1 if gender == "Male" else 0],
            'bmi': [bmi],
            'glucose': [glucose],
            'family_diabetes': [1 if family_diabetes == "Yes" else 0],
            'hypertensive': [1 if hypertensive == "Yes" else 0]
        })

        for col in model_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[model_columns]

        # Predict
        new_data_scaled = scaler.transform(new_data)
        pred = model.predict(new_data_scaled)[0]
        prob = model.predict_proba(new_data_scaled)[0][1]

        # Determine risk level and advice
        if prob < 0.33:
            risk_level = "Low"
            advice = """✅ Keep monitoring glucose every 6 months
✅ Maintain healthy BMI
✅ Follow balanced diet
✅ Exercise 30 min daily
✅ Reduce added sugar
✅ Avoid smoking and alcohol"""
        elif prob < 0.66:
            risk_level = "Medium"
            advice = """⚠️ Check blood sugar regularly
⚠️ Reduce simple carbs
⚠️ Increase protein & fiber
⚠️ Moderate exercise 4-5x/week
⚠️ Monitor BP and cholesterol
⚠️ Consult a nutritionist"""
        else:
            risk_level = "High"
            advice = """🚨 See your doctor immediately
🚨 Strict diabetic diet
🚨 Daily glucose monitoring
🚨 Regular exercise
🚨 Maintain healthy weight
🚨 Follow medical appointments"""

        # Prepare patient info
        patient_info = {
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Weight": weight,
            "Height": height,
            "BMI": bmi,
            "Glucose": glucose,
            "Risk Level": risk_level,
            "Advice": advice
        }

        # Generate PDF
        pdf_bytes = generate_pdf(patient_info)

        st.success("✅ PDF Generated Successfully!")

        # Download button outside form
        st.download_button(
            label="⬇️ Download Report",
            data=pdf_bytes,
            file_name=f"{name}_Medical_Report.pdf",
            mime="application/pdf"
        )




# =================== BATCH ANALYSIS ===================
elif page == "التحليل الجماعي | Batch Analysis":
    st.header("📊 التحليل الجماعي | Batch Analysis")

    uploaded_file = st.file_uploader(
        "اختر الملف | Choose File", type=["csv", "xlsx"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(
            ".csv") else pd.read_excel(uploaded_file)
        st.success("✅ تم تحميل الملف بنجاح | File Uploaded Successfully")

        results = []

        for _, row in df.iterrows():
            bmi = row['weight'] / ((row['height'] / 100) ** 2)
            new_data = pd.DataFrame({
                'age': [row['age']],
                'gender': [row['gender']],
                'bmi': [bmi],
                'glucose': [row['glucose']],
                'family_diabetes': [1 if str(row['family_diabetes']).endswith("Yes") else 0],
                'hypertensive': [1 if str(row['hypertensive']).endswith("Yes") else 0]
            })
            new_data = pd.get_dummies(new_data, drop_first=True)
            for col in model_columns:
                if col not in new_data.columns:
                    new_data[col] = 0
            new_data = new_data[model_columns]

            pred = model.predict(scaler.transform(new_data))[0]
            status_table = "مصاب | Diabetic" if pred == 1 else "غير مصاب | Non-Diabetic"
            arabic_status = arabic_reshaper.reshape(
                "مصاب") if pred == 1 else arabic_reshaper.reshape("غير مصاب")
            status_chart = f"{get_display(arabic_status)} | {'Diabetic' if pred == 1 else 'Non-Diabetic'}"
            results.append({
                "الاسم | Name": row['name'],
                "الحالة | Status": status_table,
                "status_chart": status_chart
            })

        results_df = pd.DataFrame(results)
        st.subheader("📋 نتائج التحليل | Batch Results")
        st.dataframe(results_df[["الاسم | Name", "الحالة | Status"]])

        status_counts = results_df['status_chart'].value_counts()
        st.subheader("📈 توزيع حالات المرضى | Patients Status Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#e74c3c', '#2ecc71']
        ax.bar(status_counts.index, status_counts.values, color=colors)
        ax.set_xlabel(get_display(
            arabic_reshaper.reshape("الحالة")) + " | Status")
        ax.set_ylabel(get_display(arabic_reshaper.reshape(
            "عدد المرضى")) + " | Number of Patients")
        ax.set_title(get_display(arabic_reshaper.reshape(
            "توزيع حالات السكري")) + " | Diabetes Status Distribution")
        for i, v in enumerate(status_counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

        st.pyplot(fig)

# =================== MEDICAL IMAGE ANALYSIS ===================
elif page == "تحليل صورة الفحص الطبي | Medical Image Analysis":
    st.header("🧪 تشخيص السكري بناءً على الجلوكوز | Diabetes Diagnosis by Glucose")

    uploaded_file = st.file_uploader("اختر صورة الفحص | Upload Image", ["png", "jpg", "jpeg"])

    if uploaded_file:
        # عرض الصورة
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        # حفظ الصورة مؤقتًا للـ OCR
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            img_path = tmp.name

        # ------------------- OCR -------------------
        reader = easyocr.Reader(['ar', 'en'])
        results = reader.readtext(img_path)

        # دمج كل النصوص
        full_text = " ".join([r[1] for r in results])

        # -------------------Glucose extraction -------------------
        glucose = None
        glucose_patterns = [
            r'Glucose[:\s]*([0-9]{2,3})',
            r'([0-9]{2,3})\s*mg\s*/?\s*dL',
            r'([0-9]{2,3})\s*mg\s*DL',
            r'سكر[:\s]*([0-9]{2,3})'  
        ]
        for p in glucose_patterns:
            match = re.search(p, full_text, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                if 50 <= value <= 500:
                    glucose = float(value)
                    break

        # ------------------- Diagnosis  -------------------
        if glucose is None:
            st.error("❌ لم يتم التعرف على قيمة الجلوكوز في الصورة | Glucose value not detected")
        else:
            st.write(f"🩸 Glucose: {glucose} mg/dL")

            if glucose < 70:
                st.warning("🔹 الجلوكوز منخفض | Low Glucose")
                st.info("""
    ✅ توصيات | Recommendations:
    - تناول وجبة صغيرة تحتوي على سكريات طبيعية | Eat a small meal with natural sugars
    - مراقبة مستوى السكر بانتظام | Monitor glucose regularly
    - مراجعة طبيب عند الحاجة | Consult a doctor if necessary
    """)
            elif 70 <= glucose <= 140:
                st.success("🟢 طبيعي | Normal | Non-Diabetic")
                st.info("""
    ✅ توصيات وقائية | Preventive Recommendations:
    - الحفاظ على نمط حياة صحي | Maintain a healthy lifestyle
    - متابعة تحليل السكر بشكل دوري | Monitor glucose periodically
    - تناول غذاء متوازن | Eat a balanced diet
    """)
            else:
                st.error("🔴 مرتفع | High Glucose: Possible Diabetes")
                st.warning("""
    🚨 توصيات طبية | Medical Recommendations:
    - مراجعة طبيب غدد فوراً | See an endocrinologist immediately
    - الالتزام بحمية لمرضى السكري | Follow a diabetic diet
    - مراقبة السكر يومياً | Monitor glucose daily
    - ممارسة الرياضة بانتظام | Exercise regularly
    """)
