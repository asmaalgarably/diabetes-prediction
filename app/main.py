import easyocr
import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import os
import pytesseract
from PIL import Image
import re
import cv2
import numpy as np

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
font_path = os.path.join(current_dir, "Fonts", "DejaVuSans.ttf")

# ------------------- Initialize session_state -------------------
if "saved_advice" not in st.session_state:
    st.session_state.saved_advice = ""

# ------------------- PDF Generation Functions -------------------


def add_arabic(pdf, text, cell_width=180, font_size=12):
    pdf.set_font("DejaVu", "", font_size)
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    pdf.multi_cell(cell_width, 8, bidi_text, align='R')
    pdf.ln(2)


def generate_pdf(patient_info):
    pdf = FPDF()
    pdf.add_page()

    # Arabic line
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"TTF Font file not found: {font_path}")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 12)

    # Logo
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=8, w=30)
        pdf.ln(25)

    # Patient Report
    add_arabic(pdf, "تقرير المريض | Patient Report", font_size=16)

    # Patient Data
    for key, value in patient_info.items():
        add_arabic(pdf, f"{key}: {value}")

    # BMI
    plt.figure(figsize=(4, 2))
    bmi_value = patient_info.get('BMI', 0)
    plt.bar(['BMI'], [bmi_value], color='blue')
    plt.axhline(y=18.5, color='green', linestyle='--', label='Underweight')
    plt.axhline(y=24.9, color='yellow', linestyle='--', label='Normal')
    plt.axhline(y=29.9, color='orange', linestyle='--', label='Overweight')
    plt.axhline(y=30, color='red', linestyle='--', label='Obese')
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='PNG')
    plt.close()
    buf.seek(0)
    pdf.image(buf, x=10, y=None, w=pdf.w - 20)
    buf.close()

    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title="نظام تقييم خطر السكري | Diabetes Risk Assessment", layout="wide", page_icon='💉🩺 ')
st.sidebar.title("التنقل | Navigation")
page = st.sidebar.radio("اذهب إلى | Go to", [
    "الرئيسية | Home",
    "تنبؤ خطر المريض | Predict Patient Risk",
    "تقرير المريض | Patient Report",
    "تحميل جماعي | Batch Upload",
    "تحليل صورة التحليل | Image Analysis"
])

# ------------------- Home Page -------------------
if page == "الرئيسية | Home":
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.warning("لم يتم العثور على شعار المستشفى | Hospital logo not found")
    st.title("💉🩺  نظام تقييم خطر السكري | Diabetes Risk Assessment System")
    st.markdown("""
    مرحباً بكم في نظام تقييم خطر السكري. | Welcome to the Diabetes Risk Assessment System.  
    هذا النظام يتنبأ بخطر السكري بناءً على بيانات المريض ويقدم توصيات صحية.  
    This system predicts the risk of diabetes based on patient data and provides health recommendations.
    """)

# ------------------- Predict Patient Risk -------------------
elif page == "تنبؤ خطر المريض | Predict Patient Risk":
    st.header("معلومات المريض | Patient Information")
    with st.form("patient_form"):
        name = st.text_input("اسم المريض (اختياري) | Patient Name (Optional)")
        age = st.number_input("العمر | Age", min_value=1,
                              max_value=120, value=35)
        gender = st.selectbox(
            "الجنس | Gender", ["ذكر | Male", "أنثى | Female"])
        weight = st.number_input(
            "الوزن (كغ) | Weight (kg)", min_value=1, max_value=300, value=70)
        height = st.number_input(
            "الطول (سم) | Height (cm)", min_value=50, max_value=250, value=170)
        glucose = st.number_input(
            "مستوى الجلوكوز (ملغ/دل) | Glucose Level (mg/dL)", min_value=50, max_value=400, value=110)
        hypertensive = st.selectbox("ارتفاع ضغط الدم؟ | Hypertension?", [
                                    "لا | No", "نعم | Yes"])
        family_diabetes = st.selectbox("سكري في العائلة؟ | Family Diabetes?", [
                                       "لا | No", "نعم | Yes"])
        submitted = st.form_submit_button("تحليل الخطر | Analyze Risk")

    if submitted:
        bmi = weight / ((height/100)**2)
        new_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'bmi': [bmi],
            'glucose': [glucose],
            'family_diabetes': [1 if family_diabetes.endswith('Yes') else 0],
            'hypertensive': [1 if hypertensive.endswith('Yes') else 0]
        })
        new_data = pd.get_dummies(new_data, drop_first=True)
        for col in model_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[model_columns]
        new_data_scaled = scaler.transform(new_data)

        pred = model.predict(new_data_scaled)[0]
        prob = model.predict_proba(new_data_scaled)[0][1]

        # Determining the level of risk and delivery

        if prob < 0.33:
            risk_level = "منخفض | Low"
            advice = "مريضك في وضع جيد. حافظ على نظام غذائي صحي وممارسة الرياضة بانتظام. | Your patient is healthy. Maintain a healthy diet and regular exercise."
            color = "green"
        elif prob < 0.66:
            risk_level = "متوسط | Medium"
            advice = "راقب مستوى الجلوكوز بانتظام وحافظ على نظام غذائي متوازن. | Monitor glucose levels regularly and maintain a balanced diet."
            color = "orange"
        else:
            risk_level = "مرتفع | High"
            advice = "استشر طبيبك وراقب HbA1c بانتظام. | Consult a doctor and monitor HbA1c regularly."
            color = "red"

        st.subheader(
            f"التنبؤ | Predicted: {'مصاب بالسكري | Diabetic' if pred == 1 else 'غير مصاب | Not Diabetic'}")
        st.subheader(f"الاحتمالية | Probability: {prob*100:.2f}%")
        st.markdown(
            f"<h3 style='color:{color}'>مستوى الخطر | Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
        st.markdown(f"**التوصيات | Recommendations:** {advice}")

        #  saving session_state
        st.session_state.saved_advice = advice

        # Copy the recommendation button for the report
        if st.button("نسخ التوصية إلى التقرير | Copy Recommendation to Report"):
            st.success(
                "تم نسخ التوصية بنجاح | Recommendation copied to report!")

# ------------------- Patient Report Page -------------------
elif page == "تقرير المريض | Patient Report":
    st.header("تحميل تقرير المريض | Download Patient PDF Report")
    st.markdown(
        "يمكنك إنشاء وتحميل تقرير PDF لمريض هنا | You can generate and download a PDF report for a patient here.")

    with st.form("pdf_form"):
        name = st.text_input("اسم المريض | Patient Name")
        age = st.number_input("العمر | Age", min_value=1,
                              max_value=120, value=35)
        gender = st.selectbox(
            "الجنس | Gender", ["ذكر | Male", "أنثى | Female"])
        weight = st.number_input(
            "الوزن (كغ) | Weight (kg)", min_value=1, max_value=300, value=70)
        height = st.number_input(
            "الطول (سم) | Height (cm)", min_value=50, max_value=250, value=170)
        glucose = st.number_input(
            "مستوى الجلوكوز (ملغ/دل) | Glucose Level (mg/dL)", min_value=50, max_value=400, value=110)
        hypertensive = st.selectbox("ارتفاع ضغط الدم؟ | Hypertension?", [
                                    "لا | No", "نعم | Yes"])
        family_diabetes = st.selectbox("سكري في العائلة؟ | Family Diabetes?", [
                                       "لا | No", "نعم | Yes"])
        predicted = st.selectbox("التنبؤ | Predicted", [
                                 "مصاب بالسكري | Diabetic", "غير مصاب | Not Diabetic"])
        risk_level = st.selectbox("مستوى الخطر | Risk Level", [
                                  "منخفض | Low", "متوسط | Medium", "مرتفع | High"])

        # Show recommendation only, no modifications
        st.text_area(
            "التوصيات | Recommendations",
            st.session_state.saved_advice,
            height=100,
            disabled=True
        )

        generate = st.form_submit_button("إنشاء PDF | Generate PDF")

    if generate:
        bmi = round(weight / ((height/100)**2), 2)
        advice = st.session_state.saved_advice or "لا توجد توصية حالياً | No recommendation available."

        patient_info = {
            "الاسم | Name": name,
            "العمر | Age": age,
            "الجنس | Gender": gender,
            "الوزن | Weight (kg)": weight,
            "الطول | Height (cm)": height,
            "BMI": bmi,
            "مستوى الجلوكوز | Glucose": glucose,
            "ارتفاع ضغط الدم | Hypertension": hypertensive,
            "سكري في العائلة | Family Diabetes": family_diabetes,
            "التنبؤ | Predicted": predicted,
            "مستوى الخطر | Risk Level": risk_level,
            "التوصيات | Recommendations": advice
        }
        pdf_bytes = generate_pdf(patient_info)
        st.success("تم إنشاء التقرير بنجاح | PDF report generated successfully")
        st.download_button("تحميل تقرير PDF | Download PDF", data=pdf_bytes,
                           file_name=f"{name or 'patient'}_report.pdf", mime="application/pdf")

# ------------------- Batch Upload -------------------
elif page == "تحميل جماعي | Batch Upload":
    st.header(
        "رفع ملف المرضى وتحليل جماعي | Upload Patients Data for Batch Analysis")
    st.markdown("""
    ارفع ملف CSV أو Excel يحتوي على بيانات المرضى للتحليل. | Upload a CSV or Excel file with patient data for analysis.  
    الأعمدة المطلوبة | Required columns: `id, name (optional), age, gender, weight, height, glucose, hypertensive, family_diabetes`
    """)

    uploaded_file = st.file_uploader(
        "اختر الملف | Choose file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(
                f"تم تحميل الملف بنجاح | File loaded successfully with {len(df)} patients")

            results = []
            risk_counts = {"مصاب بالسكري | Diabetic": 0,
                           "غير مصاب | Not Diabetic": 0}
            for idx, row in df.iterrows():
                bmi = row['weight'] / ((row['height']/100)**2)
                new_data = pd.DataFrame({
                    'age': [row['age']],
                    'gender': [row['gender']],
                    'bmi': [bmi],
                    'glucose': [row['glucose']],
                    'family_diabetes': [1 if str(row['family_diabetes']).endswith('Yes') else 0],
                    'hypertensive': [1 if str(row['hypertensive']).endswith('Yes') else 0]
                })
                new_data = pd.get_dummies(new_data, drop_first=True)
                for col in model_columns:
                    if col not in new_data.columns:
                        new_data[col] = 0
                new_data = new_data[model_columns]
                new_data_scaled = scaler.transform(new_data)

                pred = model.predict(new_data_scaled)[0]
                status = 'مصاب بالسكري | Diabetic' if pred == 1 else 'غير مصاب | Not Diabetic'
                results.append({
                    "رقم المريض | Patient ID": row['id'] if 'id' in row else idx+1,
                    "الاسم | Name": row['name'] if 'name' in row else f"Patient {idx+1}",
                    "الحالة | Status": status
                })
                risk_counts[status] += 1

            results_df = pd.DataFrame(results)
            st.subheader("جدول تحليل المرضى | Patients Analysis Table")
            st.dataframe(results_df)

            # Chart
            st.subheader("توزيع المرضى | Status Distribution")
            fig, ax = plt.subplots()
            ax.bar(risk_counts.keys(), risk_counts.values(),
                   color=['red', 'green'])
            ax.set_ylabel("عدد المرضى | Number of Patients")
            st.pyplot(fig)

            # Download CSV
            csv_buffer = BytesIO()
            results_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button("تحميل نتائج التحليل CSV | Download Batch Results CSV",
                               data=csv_buffer.getvalue(), file_name="batch_results.csv", mime="text/csv")

        except Exception as e:
            st.error(
                f"حدث خطأ أثناء معالجة الملف | Error processing file: {e}")
            
# ================== Image Analysis with EasyOCR ==================

elif page == "تحليل صورة التحليل | Image Analysis":
    st.header("تحليل صورة التحليل الطبي | Medical Image Analysis")
    st.markdown(
        "ارفع صورة التحليل الطبي وسيقوم النظام بتحليلها وتقديم تقدير للحالة الصحية والتوصية."
    )

    uploaded_file = st.file_uploader(
        "اختر صورة التحليل", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        # عرض الصورة
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)

        # تهيئة EasyOCR للغة الإنجليزية والعربية
        reader = easyocr.Reader(['en', 'ar'])

        # استخراج النصوص من الصورة
        result = reader.readtext(np.array(image))

        if result:
            st.subheader("النصوص المستخرجة من الصورة:")
            for (_, text, prob) in result:
                st.write(f"- {text} (دقة: {prob:.2f})")

            # محاولة إيجاد قيمة الجلوكوز
            glucose_values = []
            for (_, text, _) in result:
                # تحويل الأرقام العربية إلى إنجليزية
                text = text.translate(
                    str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
                # البحث عن أرقام من خانتين أو ثلاث
                matches = re.findall(r'\b\d{2,3}\b', text)
                glucose_values.extend(matches)

            # تصفية القيم الواقعية للجلوكوز
            glucose_values = [
                val for val in glucose_values if 40 <= int(val) <= 500]

            # ------------------- تحليل قيم الجلوكوز -------------------
        if glucose_values:
            st.success(
                f"تم العثور على قيم محتملة للجلوكوز: {', '.join(glucose_values)}")

            st.subheader("تحليل حالة الجلوكوز:")
            for val in glucose_values:
                val_int = int(val)
                if val_int < 70:
                    st.warning(f"قيمة {val_int} mg/dL → منخفض ⚠️")
                elif 70 <= val_int <= 140:
                    st.success(f"قيمة {val_int} mg/dL → طبيعي ✅")
                else:
                    st.error(f"قيمة {val_int} mg/dL → مرتفع ❌")
        else:
            st.warning("لم يتم العثور على قيمة جلوكوز واضحة لتحليلها.")
