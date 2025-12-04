import pytesseract
import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import os
from PIL import Image
import re
import numpy as np
import tempfile

# ------------------- Load Model, Scaler, Columns -------------------
with open('../models/rf_diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/rf_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../models/rf_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# ------------------- Paths -------------------
current_dir = os.path.dirname(__file__)
logo_path = os.path.join(current_dir, "..", "image", "logo.png")
logo_path1 = os.path.join(current_dir, "..", "image", "logo1.png")
font_path = os.path.join(current_dir, "Fonts", "DejaVuSans.ttf")

# ------------------- Session State -------------------
if "saved_advice" not in st.session_state:
    st.session_state.saved_advice = ""

# ------------------- PDF Functions -------------------


def add_arabic(pdf, text, font_size=12, bold=False):
    style = "B" if bold else ""
    pdf.set_font("DejaVu", style, font_size)
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    pdf.multi_cell(0, 8, bidi_text, align='R')
    pdf.ln(2)


def generate_pdf(patient_info):
    pdf = FPDF()
    pdf.add_page()

    # ------------------ Borders------------------
    pdf.set_line_width(0.5)
    pdf.rect(5, 5, 200, 287)  

    # ------------------add lines ------------------
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.add_font("DejaVu", "B", font_path, uni=True)

    pdf.set_font("DejaVu", "", 12)

    # ------------------ Logo------------------
    if os.path.exists(logo_path):
        logo_width = 70
        page_width = pdf.w - 2 * pdf.l_margin
        x_center = (page_width - logo_width) / 2 + pdf.l_margin
        pdf.image(logo_path1, x=x_center, y=10, w=logo_width)
        pdf.ln(35)

    # ------------------ Report------------------

    add_arabic(pdf, "Patient's medical report | التقرير الطبي للمريض",
               font_size=16, bold=True)

    # ------------------ Patient data ------------------
    for key, value in patient_info.items():
        add_arabic(pdf, f"{key}: {value}")

    pdf_bytes = pdf.output(dest='S')  
    pdf_buffer = BytesIO(pdf_bytes)
    return pdf_buffer.getvalue()



# ------------------- Page Config -------------------
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

# =================== PAGE 2: SAVE PATIENT DATA ===================
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

# =================== MEDICAL REPORT PAGE ===================
elif page == "التقرير الطبي للمريض | Medical Report":

    st.header("📄 التقرير الطبي للمريض | Patient Medical Report")

    with st.form("pdf_form"):
        name = st.text_input("اسم المريض | Patient Name",
                             value=st.session_state.get("home_name", ""))
        age = st.number_input("العمر | Age", 1, 120,
                              value=st.session_state.get("home_age", 35))
        gender = st.selectbox(
            "الجنس | Gender", ["ذكر | Male", "أنثى | Female"],
            index=0 if st.session_state.get(
                "home_gender", "ذكر | Male") == "ذكر | Male" else 1
        )
        weight = st.number_input(
            "الوزن (كغ) | Weight (kg)", 1, 300, value=st.session_state.get("home_weight", 70))
        height = st.number_input("الطول (سم) | Height (cm)", 50,
                                 250, value=st.session_state.get("home_height", 170))
        glucose = st.number_input("مستوى الجلوكوز | Glucose (mg/dL)", 50, 400,
                                  value=st.session_state.get("home_glucose", 110))
        hypertensive = st.selectbox(
            "ارتفاع ضغط الدم | Hypertension", ["لا | No", "نعم | Yes"],
            index=0 if st.session_state.get(
                "home_hypertension", "لا | No") == "لا | No" else 1
        )
        family_diabetes = st.selectbox(
            "تاريخ عائلي للسكري | Family Diabetes", ["لا | No", "نعم | Yes"],
            index=0 if st.session_state.get(
                "home_family_diabetes", "لا | No") == "لا | No" else 1
        )

        save_button = st.form_submit_button(
            "💾 حفظ وإنشاء التقرير | Save & Generate PDF")

    if save_button:

        bmi = round(weight / ((height / 100) ** 2), 2)

        new_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'bmi': [bmi],
            'glucose': [glucose],
            'family_diabetes': [1 if family_diabetes.endswith("Yes") else 0],
            'hypertensive': [1 if hypertensive.endswith("Yes") else 0]
        })
        new_data = pd.get_dummies(new_data, drop_first=True)
        for col in model_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[model_columns]
        new_data_scaled = scaler.transform(new_data)

        pred = model.predict(new_data_scaled)[0]
        prob = model.predict_proba(new_data_scaled)[0][1]

        if prob < 0.33:
            risk_level = "منخفض | Low"
            advice = """
✅ الحالة جيدة، استمر بالمراقبة الدورية لمستوى الجلوكوز كل 6 أشهر.
✅ حافظ على وزن صحي ومستوى BMI مناسب.
✅ اتبع نظام غذائي متوازن غني بالخضار والفواكه والحبوب الكاملة.
✅ مارس النشاط البدني 30 دقيقة يوميًا على الأقل.
✅ قلل السكريات المضافة والمشروبات الغازية.
✅ تجنب التدخين والكحول.
"""
        elif prob < 0.66:
            risk_level = "متوسط | Medium"
            advice = """
⚠️ قياس السكر في الدم بانتظام، على الأقل مرة أسبوعيًا.
⚠️ قلل الكربوهيدرات البسيطة (الخبز الأبيض، الحلويات).
⚠️ زد من استهلاك البروتينات الصحية والخضروات الغنية بالألياف.
⚠️ مارس الرياضة معتدلة الشدة 4-5 مرات أسبوعيًا.
⚠️ راقب ضغط الدم والكوليسترول بانتظام.
⚠️ استشر أخصائي تغذية لتخطيط نظام غذائي شخصي.
"""
        else:
            risk_level = "مرتفع | High"
            advice = """
🚨 راجع طبيبك فورًا لإجراء فحوصات شاملة (سكر، HbA1c، ضغط الدم).
🚨 ضع خطة علاجية مخصصة إذا تم تشخيصك بالسكري.
🚨 قلل السكريات والكربوهيدرات المكررة بشكل صارم.
🚨 مارس تمارين يومية معتدلة، وتجنب الخمول.
🚨 حافظ على وزن صحي وقلل الدهون المشبعة.
🚨 راقب الجلوكوز في الدم يوميًا إذا كنت مصابًا.
🚨 التزم بالمواعيد الدورية للطبيب وأخصائي التغذية.
"""

        # إنشاء PDF
        patient_info = {
            "الاسم | Name": name,
            "العمر | Age": age,
            "الجنس | Gender": gender,
            "الوزن | Weight": weight,
            "الطول | Height": height,
            "BMI": bmi,
            "مستوى الجلوكوز | Glucose": glucose,
            "مستوى الخطورة | Risk Level": risk_level
        }

        pdf_bytes = generate_pdf({**patient_info, "التوصيات الطبية": advice})

        st.success("✅ تم إنشاء التقرير بنجاح | PDF Generated Successfully")
        st.download_button("⬇️ تحميل التقرير | Download Report",
                           pdf_bytes, f"{name}_Medical_Report.pdf")

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


# ------------------- images Analysis -------------------
elif page == "تحليل صورة الفحص الطبي | Medical Image Analysis":
    st.header("🧪 تشخيص السكري بناءً على الجلوكوز | Diabetes Diagnosis by Glucose")

    uploaded_file = st.file_uploader(
        "اختر صورة الفحص | Upload Image", ["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        # ------------------- OCR using Tesseract -------------------
        extracted_text = pytesseract.image_to_string(image, lang="eng")

        # ------------------- Glucose extraction -------------------
        glucose = None
        glucose_patterns = [
            r'Glucose[:\s]*([0-9]{2,3})',
            r'([0-9]{2,3})\s*mg\s*/?\s*dL',
            r'([0-9]{2,3})\s*mg\s*DL',
            r'سكر[:\s]*([0-9]{2,3})'
        ]

        for p in glucose_patterns:
            match = re.search(p, extracted_text, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                if 50 <= value <= 500:
                    glucose = float(value)
                    break

        # ------------------- Diagnosis -------------------
        if glucose is None:
            st.error(
                "❌ لم يتم التعرف على قيمة الجلوكوز في الصورة | Glucose value not detected"
            )
        else:
            st.write(f"🩸 Glucose: {glucose} mg/dL")

            if glucose < 70:
                st.warning("🔹 الجلوكوز منخفض | Low Glucose")
                st.info("""
✅ توصيات | Recommendations:
- تناول وجبة تحتوي على سكريات طبيعية | Eat natural sugars
- متابعة مستوى السكر | Monitor glucose
- مراجعة الطبيب عند الحاجة | Consult a doctor
""")

            elif 70 <= glucose <= 140:
                st.success("🟢 طبيعي | Normal | Non-Diabetic")
                st.info("""
✅ توصيات وقائية | Preventive Recommendations:
- نمط حياة صحي | Healthy lifestyle
- فحص دوري للسكر | Periodic glucose check
- غذاء متوازن | Balanced diet
""")

            else:
                st.error("🔴 مرتفع | High Glucose: Possible Diabetes")
                st.warning("""
🚨 توصيات طبية | Medical Recommendations:
- مراجعة طبيب فورًا | See a doctor immediately
- حمية لمرضى السكري | Diabetic diet
- فحص السكر يوميًا | Daily glucose check
- ممارسة الرياضة | Exercise regularly
""")
