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


# 2. تعريف دالة التحميل مع التخزين المؤقت للسرعة

@st.cache_resource
def load_ocr_reader():
    # التحميل يتم مرة واحدة فقط عند تشغيل التطبيق
    return easyocr.Reader(['ar', 'en'], gpu=False)


# 3. تعريف المتغير 'reader' بشكل عالمي (Global)
reader = load_ocr_reader()


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
    pdf.cell(0, 10, "This report supports medical decisions and does not replace consultation with a physician.", 0, 0, 'C'
             )
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin-1')
    elif isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)

    return pdf_bytes


# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title="النظام الذكي لتقييم خطر الإصابة بالسكري | Smart Diabetes Risk Assessment",
    layout="wide",
    page_icon="💉"
)
st.markdown("""
    <style>
    /* تحسين الخطوط والخلفية */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    .stApp { background-color: #0b0e14; }

    /* الهيدر الرئيسي بتأثير زجاجي متطور */
    .hero-gradient {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.9) 0%, rgba(59, 130, 246, 0.8) 100%);
        padding: 60px 20px;
        border-radius: 30px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 50px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }

    /* كروت الخدمات مع إضاءة علوية */
    .feature-card-modern {
        background: #161b22;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #30363d;
        border-top: 4px solid #3b82f6; /* خط علوي ملون */
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
    }
    .feature-card-modern:hover {
        transform: translateY(-12px);
        box-shadow: 0 15px 30px rgba(59, 130, 246, 0.2);
        border-color: #3b82f6;
    }

    .icon-circle {
        width: 70px;
        height: 70px;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px;
        font-size: 30px;
    }

    /* تنسيق حاوية النموذج */
    .form-container {
        background: #161b22;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #30363d;
        margin-top: 20px;
    }
    /* جعل العناوين الجانبية للمدخلات أوضح */
    label {
        color: #58a6ff !important;
        font-weight: bold !important;
    }

    .feature-title { color: #58a6ff; font-weight: bold; margin-bottom: 10px; }
    .feature-desc { color: #8b949e; font-size: 0.9rem; line-height: 1.5; }
    </style>
    """, unsafe_allow_html=True)

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
    # عرض اللوجو بشكل أنيق
    if os.path.exists(logo_path):
        st.columns([1, 1.2, 1])[1].image(logo_path, use_container_width=True)

    # قسم الترحيب الرئيسي
    st.markdown("""
        <div class="hero-gradient">
            <h1 style="font-family: 'Cairo', sans-serif; font-size: 3.5rem; margin-bottom: 0;">Smart Diabetes AI</h1>
            <p style="font-size: 1.4rem; font-weight: 300; letter-spacing: 1px;">النظام الذكي المتكامل للوقاية من السكري</p>
            <div style="width: 100px; height: 3px; background: #58a6ff; margin: 20px auto;"></div>
            <p style="font-size: 1rem; opacity: 0.8; max-width: 700px; margin: 0 auto;">
                نجمع بين دقة البيانات وقوة الذكاء الاصطناعي لنقدم لك تحليلاً طبياً استباقياً يساعدك في اتخاذ قراراتك الصحية بثقة.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # عرض المميزات في شبكة احترافية
    st.markdown("<h3 style='text-align: center; color: white; margin-bottom: 40px;'>مميزات المنصة | Platform Features</h3>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    features = [
        {"icon": "🧬", "title": "تنبؤ دقيق", "title_en": "AI Prediction",
            "desc": "تحليل المخاطر باستخدام خوارزميات متقدمة."},
        {"icon": "📋", "title": "تقارير شاملة", "title_en": "Medical Reports",
            "desc": "توليد ملفات PDF تحتوي على كافة التفاصيل."},
        {"icon": "🔍", "title": "قراءة الصور", "title_en": "Medical OCR",
            "desc": "استخراج النتائج من صور الفحوصات الطبية."},
        {"icon": "📊", "title": "تحليل الدفعات", "title_en": "Batch Analysis",
            "desc": "معالجة بيانات مجموعة مرضى دفعة واحدة."}
    ]

    cols = [col1, col2, col3, col4]
    for i, f in enumerate(features):
        with cols[i]:
            st.markdown(f"""
                <div class="feature-card-modern">
                    <div class="icon-circle">{f['icon']}</div>
                    <div class="feature-title">{f['title']}</div>
                    <div style="color: white; font-size: 0.8rem; opacity: 0.6; margin-bottom: 10px;">{f['title_en']}</div>
                    <p class="feature-desc">{f['desc']}</p>
                </div>
            """, unsafe_allow_html=True)

    # تذييل الصفحة وتنبيه الطبي
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.error("⚠️ **ملاحظة هامة:** النتائج المقدمة من النظام هي لغرض الدعم المعلوماتي فقط ولا تعتبر تشخيصاً نهائياً. يرجى دائماً استشارة الطبيب المختص.")

# =================== PAGE 2: PATIENT RISK ASSESSMENT ===================
# --- صفحة تقييم خطر المريض المحدثة ---
elif page == "تقييم خطر المريض | Patient Risk Assessment":
    st.markdown("""
        <div style="text-align: center; margin-bottom: 25px;">
            <h2 style="color: #3b82f6; font-family: 'Cairo', sans-serif;">🧑‍⚕️ نموذج تقييم بيانات المريض</h2>
            <p style="color: #8b949e;">يرجى إدخال البيانات الحيوية بدقة لضمان دقة التنبؤ</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form_modern"):
        # الجزء الأول: المعلومات الشخصية
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("اسم المريض بالكامل",
                                 value=st.session_state.get("home_name", ""))

            # التأكد من أن العمر رقم صحيح (int)
            age_val = int(st.session_state.get("home_age", 35))
            age = st.number_input("العمر (سنة)", min_value=1,
                                  max_value=120, value=age_val)

        with col2:
            gender = st.selectbox("الجنس", ["Male", "Female"],
                                  index=0 if st.session_state.get("home_gender") == "Male" else 1)

            # حل مشكلة الخطأ: تحويل الجلوكوز القادم من الـ OCR إلى int فوراً
            glucose_val = int(float(st.session_state.get("home_glucose", 110)))
            glucose = st.number_input(
                "مستوى الجلوكوز الصائم (mg/dL)", min_value=50, max_value=400, value=glucose_val)

        st.markdown("<hr style='border-color: #30363d;'>",
                    unsafe_allow_html=True)

        # الجزء الثاني: القياسات الحيوية والتاريخ الطبي
        col3, col4 = st.columns(2)
        with col3:
            weight = st.number_input("الوزن (كجم)", min_value=1, max_value=300, value=int(
                st.session_state.get("home_weight", 70)))
            height = st.number_input("الطول (سم)", min_value=50, max_value=250, value=int(
                st.session_state.get("home_height", 170)))

        with col4:
            hyper = st.selectbox("هل يعاني من ارتفاع ضغط الدم؟", ["No", "Yes"],
                                 index=1 if st.session_state.get("home_hypertension") == "Yes" else 0)
            family = st.selectbox("هل يوجد تاريخ عائلي للسكري؟", ["No", "Yes"],
                                  index=1 if st.session_state.get("home_family_diabetes") == "Yes" else 0)

        # زر الحفظ
        submit_button = st.form_submit_button(
            "💾 حفظ البيانات وتحليلها", use_container_width=True)

    if submit_button:
        # حفظ كل البيانات في الـ Session State لاستخدامها في صفحة التقرير
        st.session_state.update({
            "home_name": name,
            "home_age": age,
            "home_gender": gender,
            "home_glucose": glucose,
            "home_weight": weight,
            "home_height": height,
            "home_hypertension": hyper,
            "home_family_diabetes": family
        })
        st.success(f"✅ تم تحديث بيانات المريض {name} بنجاح!")
        st.info("💡 يمكنك الآن الانتقال إلى صفحة 'التقرير الطبي' لمشاهدة تحليل النتائج.")
# =================== PAGE 3: MEDICAL REPORT ===================
elif page == "التقرير الطبي للمريض | Medical Report":
    st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🏥 لوحة تشخيص حالة المريض | Medical Dashboard</h2>", unsafe_allow_html=True)
    st.write("")

    # تقسيم الصفحة لعمودين: يسار للمدخلات ويمين للنتائج لملء الفراغ
    col_input, col_result = st.columns([0.8, 2], gap="large")

    with col_input:
        st.markdown("#### ⚙️ التحكم والمراجعة")
        with st.form("medical_report_form"):
            # جلب البيانات المخزنة من صفحة الإدخال
            name = st.session_state.get("home_name", "Unknown")
            age = st.session_state.get("home_age", 0)
            glucose = st.session_state.get("home_glucose", 0)
            weight = st.session_state.get("home_weight", 0)
            height = st.session_state.get("home_height", 1)
            gender = st.session_state.get("home_gender", "Male")
            hypertensive = st.session_state.get("home_hypertension", "No")
            family_diabetes = st.session_state.get(
                "home_family_diabetes", "No")

            st.write(f"👤 **المريض:** {name}")
            st.write(f"🩸 **الجلوكوز:** {glucose} mg/dL")
            st.markdown("---")

            submit = st.form_submit_button(
                "📊 تحديث وتحليل النتيجة", use_container_width=True)

        # نصيحة جانبية لملء المساحة
        st.markdown("""
            <div style="background: #161b22; padding: 15px; border-radius: 12px; font-size: 0.85rem; border-left: 3px solid #3b82f6; color: #8b949e;">
                💡 <b>نظام دعم القرار:</b> هذا التحليل يعتمد على نموذج Random Forest بدقة عالية للتنبؤ بمخاطر السكري.
            </div>
        """, unsafe_allow_html=True)

    with col_result:
        if not submit:
            # شكل جمالي عند فتح الصفحة لأول مرة
            st.markdown("""
                <div style="text-align: center; padding: 100px 20px; border: 2px dashed #30363d; border-radius: 30px; background: #0d1117;">
                    <div style="font-size: 60px; margin-bottom: 20px;">🔍</div>
                    <h3 style="color: #8b949e;">جاهز للتحليل العميـق</h3>
                    <p style="color: #484f58;">الرجاء الضغط على زر "تحديث وتحليل النتيجة" لمعالجة البيانات وعرض التقرير</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # 1. الحسابات والذكاء الاصطناعي
            bmi = round(weight / ((height / 100) ** 2), 2)

            new_data = pd.DataFrame({
                'age': [age],
                'gender': [1 if gender == "Male" else 0],
                'bmi': [bmi],
                'glucose': [glucose],
                'family_diabetes': [1 if "نعم" in family_diabetes or "Yes" in family_diabetes else 0],
                'hypertensive': [1 if "نعم" in hypertensive or "Yes" in hypertensive else 0]
            })

            for col in model_columns:
                if col not in new_data.columns:
                    new_data[col] = 0

            new_data_scaled = scaler.transform(new_data[model_columns])
            prob = model.predict_proba(new_data_scaled)[0][1]

            # 2. تحديد الحالة والألوان والتوصيات (لحل مشكلة NameError)
            if prob < 0.33:
                risk_level, color, icon = "Low", "#2ecc71", "🟢"
                advice = "✅ Keep monitoring glucose every 6 months\n✅ Maintain healthy BMI\n✅ Follow balanced diet\n✅ Exercise 30 min daily"
            elif prob < 0.66:
                risk_level, color, icon = "Medium", "#f39c12", "🟡"
                advice = "⚠️ Check blood sugar regularly\n⚠️ Reduce simple carbs\n⚠️ Increase protein & fiber\n⚠️ Moderate exercise 4-5x/week"
            else:
                risk_level, color, icon = "High", "#e74c3c", "🔴"
                advice = "🚨 See your doctor immediately\n🚨 Strict diabetic diet\n🚨 Daily glucose monitoring\n🚨 Regular exercise"

            # ------------------- العرض البصري (UI) -------------------

            # كارد النتيجة الرئيسي (Gauge)
            st.markdown(f"""
                <div style="background: {color}15; border: 1px solid {color}; padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 25px;">
                    <h4 style="margin: 0; color: {color}; text-transform: uppercase; letter-spacing: 2px;">{icon} Risk Level: {risk_level}</h4>
                    <div style="width: 140px; height: 140px; border-radius: 50%; border: 8px solid {color}; display: flex; align-items: center; justify-content: center; margin: 20px auto; font-size: 28px; font-weight: bold; color: white; box-shadow: 0 0 15px {color}44;">
                        {prob*100:.1f}%
                    </div>
                    <p style="color: #8b949e; margin: 0;">احتمالية الإصابة بناءً على التحليل الرقمي</p>
                </div>
            """, unsafe_allow_html=True)

            # كروت البيانات الحيوية
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-box-v2"><h5>BMI</h5><h2 style="color:#58a6ff;">{bmi}</h2><small>مؤشر الكتلة</small></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(
                    f'<div class="metric-box-v2"><h5>Glucose</h5><h2 style="color:#58a6ff;">{glucose}</h2><small>mg/dL</small></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(
                    f'<div class="metric-box-v2"><h5>Age</h5><h2 style="color:#58a6ff;">{age}</h2><small>سنة</small></div>', unsafe_allow_html=True)

            # التوصيات الطبية
            st.markdown("### 📋 التوصيات الطبية المخصصة")
            advice_html = "".join(
                [f"<div style='margin-bottom:10px; font-size: 1.05rem;'>• {line}</div>" for line in advice.split('\n')])
            st.markdown(f"""
                <div style="background: #161b22; padding: 25px; border-radius: 15px; border-right: 5px solid {color}; color: #e6edf3; line-height: 1.6;">
                    {advice_html}
                </div>
            """, unsafe_allow_html=True)

            # زر التحميل
            st.write("---")
            patient_info = {"Name": name, "Age": age, "Gender": gender, "Weight": weight,
                            "Height": height, "BMI": bmi, "Glucose": glucose, "Risk Level": risk_level, "Advice": advice}
            pdf_bytes = generate_pdf(patient_info)
            st.download_button(label="📥 تحميل التقرير الطبي الكامل (PDF)", data=pdf_bytes,
                               file_name=f"Report_{name}.pdf", mime="application/pdf", use_container_width=True)
# =================== BATCH ANALYSIS ===================
elif page == "التحليل الجماعي | Batch Analysis":
    st.markdown("<h2 style='text-align: center;'>📊 التحليل الجماعي والبيانات الضخمة</h2>",
                unsafe_allow_html=True)

    # منطقة رفع الملفات بتصميم جذاب
    st.markdown("""
        <div style="background: #161b22; padding: 20px; border-radius: 15px; border: 1px dashed #3b82f6; text-align: center; margin-bottom: 25px;">
            <p style="margin-bottom: 10px; color: #8b949e;">يرجى رفع ملف Excel أو CSV يحتوي على بيانات المرضى (الاسم، العمر، الطول، الوزن، الجلوكوز...)</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(
            ".csv") else pd.read_excel(uploaded_file)
        st.success(f"✅ تم تحميل الملف بنجاح! يحتوي على {len(df)} مريض.")

        results = []
        # معالجة البيانات
        for _, row in df.iterrows():
            bmi = round(row['weight'] / ((row['height'] / 100) ** 2), 2)
            new_data = pd.DataFrame({
                'age': [row['age']], 'gender': [1 if str(row['gender']).lower() == "male" else 0],
                'bmi': [bmi], 'glucose': [row['glucose']],
                'family_diabetes': [1 if "Yes" in str(row['family_diabetes']) else 0],
                'hypertensive': [1 if "Yes" in str(row['hypertensive']) else 0]
            })

            for col in model_columns:
                if col not in new_data.columns:
                    new_data[col] = 0

            # التنبؤ بالاحتمالية وليس فقط التصنيف
            prob = model.predict_proba(
                scaler.transform(new_data[model_columns]))[0][1]
            risk = "High" if prob > 0.66 else "Medium" if prob > 0.33 else "Low"

            results.append({
                "Patient Name": row['name'],
                "Glucose": row['glucose'],
                "BMI": bmi,
                "Risk Probability": f"{prob*100:.1f}%",
                "Status": risk
            })

        results_df = pd.DataFrame(results)

        # --- 🟢 قسم الإحصائيات العلوية ---
        st.markdown("### 📈 ملخص الدفعة | Batch Summary")
        k1, k2, k3, k4 = st.columns(4)

        total_patients = len(results_df)
        high_risk_count = len(results_df[results_df['Status'] == "High"])
        avg_glucose = round(results_df['Glucose'].mean(), 1)

        k1.markdown(
            f'<div class="metric-box-v2"><h5>إجمالي المرضى</h5><h2 style="color:#58a6ff;">{total_patients}</h2></div>', unsafe_allow_html=True)
        k2.markdown(
            f'<div class="metric-box-v2"><h5>مخاطر عالية</h5><h2 style="color:#e74c3c;">{high_risk_count}</h2></div>', unsafe_allow_html=True)
        k3.markdown(
            f'<div class="metric-box-v2"><h5>متوسط الجلوكوز</h5><h2 style="color:#f39c12;">{avg_glucose}</h2></div>', unsafe_allow_html=True)
        k4.markdown(
            f'<div class="metric-box-v2"><h5>نسبة الخطورة</h5><h2 style="color:#2ecc71;">{round((high_risk_count/total_patients)*100)}%</h2></div>', unsafe_allow_html=True)

        st.write("---")

        # --- 🔵 عرض الجدول والمخطط جنباً إلى جنب ---
        col_tab, col_chart = st.columns([1.5, 1])

        with col_tab:
            st.markdown("#### 📋 تفاصيل المرضى")
            # تلوين الجدول بناءً على الحالة

            def color_risk(val):
                color = '#e74c3c' if val == "High" else '#f39c12' if val == "Medium" else '#2ecc71'
                return f'color: {color}; font-weight: bold'

            st.dataframe(results_df.style.applymap(
                color_risk, subset=['Status']), use_container_width=True)

        with col_chart:
            st.markdown("#### 📊 توزيع المخاطر")
            status_counts = results_df['Status'].value_counts()

            # تحسين المخطط البياني
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('#0d1117')  # لون خلفية الداشبورد
            ax.set_facecolor('#0d1117')

            colors_map = {'High': '#e74c3c',
                          'Medium': '#f39c12', 'Low': '#2ecc71'}
            ordered_colors = [colors_map.get(
                x, '#58a6ff') for x in status_counts.index]

            status_counts.plot(kind='pie', autopct='%1.1f%%',
                               colors=ordered_colors, ax=ax, textprops={'color': "w"})
            ax.set_ylabel('')
            st.pyplot(fig)

        # --- 🟡 خيار تحميل النتائج ---
        st.write("---")
        csv_results = results_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 تحميل نتائج التحليل بالكامل (CSV)",
            data=csv_results,
            file_name="Batch_Analysis_Results.csv",
            mime="text/csv",
            use_container_width=True
        )

# =================== MEDICAL IMAGE ANALYSIS ===================
elif page == "تحليل صورة الفحص الطبي | Medical Image Analysis":
    # 1. العنوان والوصف بتصميم جذاب
    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #58a6ff; font-family: 'Cairo', sans-serif;">🧪 المختبر الذكي | AI Image Lab</h1>
            <p style="color: #8b949e; font-size: 1.1rem;">استخدم الذكاء الاصطناعي لاستخراج نتائج التحاليل من الصور بدقة وسرعة</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. منطقة رفع الملف بتصميم مخصص
    st.markdown('<p style="text-align: right; color: #58a6ff;">⬇️ ارفع صورة الفحص الطبية هنا (JPG, PNG, JPEG)</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # تقسيم الصفحة لعمودين (الصورة على اليسار والنتائج على اليمين)
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            # عرض الصورة المرفوعة داخل إطار أنيق
            image = Image.open(uploaded_file)
            st.markdown(
                '<div style="border: 2px solid #30363d; border-radius: 15px; padding: 10px; background: #0d1117;">', unsafe_allow_html=True)
            st.image(image, caption="الصورة الأصلية للمختبر",
                     use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            # حاوية النتائج والتحليل
            with st.status("🔍 جاري فحص الصورة واستخراج النصوص...", expanded=True) as status:
                # حفظ مؤقت للصورة للقيام بعملية الـ OCR
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    image.save(tmp.name)
                    img_path = tmp.name

                # إجراء المسح الضوئي (باستخدام التخزين المؤقت المسبق للسرعة)
                results = reader.readtext(img_path)
                full_text = " ".join([r[1] for r in results])

                # البحث عن قيمة الجلوكوز بأنماط Regex متطورة
                glucose = None
                glucose_patterns = [
                    r'Glucose[:\s]*([0-9]{2,3})',
                    r'([0-9]{2,3})\s*mg\s*/?\s*dL',
                    r'سكر[:\s]*([0-9]{2,3})',
                    r'Result[:\s]*([0-9]{2,3})',
                    r'([0-9]{2,3})\s*Milligrams'
                ]
                for p in glucose_patterns:
                    match = re.search(p, full_text, re.IGNORECASE)
                    if match:
                        value = int(match.group(1))
                        if 50 <= value <= 500:  # نطاق منطقي للتحقق
                            glucose = float(value)
                            break

                status.update(label="✅ اكتمل استخراج البيانات بنجاح!",
                              state="complete", expanded=False)

            # --- عرض النتائج النهائية ---
            if glucose is None:
                st.error("❌ عذراً، لم نتمكن من تحديد قيمة الجلوكوز في هذه الصورة.")
                st.warning(
                    "💡 نصيحة: تأكد من أن الصورة واضحة، ليست مهتزة، وأن الإضاءة جيدة حول رقم النتيجة.")
            else:
                # كرت عرض القيمة الرقمية المكتشفة
                st.markdown(f"""
                    <div style="background: #1c2128; padding: 30px; border-radius: 20px; border-top: 6px solid #3b82f6; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                        <p style="color: #8b949e; font-size: 1rem; margin-bottom: 5px;">القيمة المكتشفة (الجلوكوز)</p>
                        <h1 style="color: #ffffff; font-size: 4.5rem; margin: 0; font-family: 'Courier New';">{int(glucose)}</h1>
                        <p style="color: #3b82f6; font-size: 1.2rem; font-weight: bold;">mg/dL</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # تشخيص سريع بناءً على الرقم المستخرج
                if glucose < 70:
                    status_msg = "⚠️ انخفاض في السكر"
                    status_color = "#f1c40f"  # أصفر
                    advice = "يُنصح بتناول مصدر سريع للسكر (عصير أو حبة تمر) وإعادة الفحص."
                elif 70 <= glucose <= 140:
                    status_msg = "✅ مستوى طبيعي"
                    status_color = "#2ecc71"  # أخضر
                    advice = "النتيجة ضمن النطاق الطبيعي للصيام أو بعد الأكل بفترة وجيزة."
                else:
                    status_msg = "🚨 مستوى مرتفع"
                    status_color = "#e74c3c"  # أحمر
                    advice = "هذه القيمة تشير لارتفاع السكر. يرجى مراجعة الطبيب لعمل فحص HbA1c."

                # عرض التشخيص
                st.markdown(f"""
                    <div style="background: {status_color}20; border-right: 5px solid {status_color}; padding: 15px; border-radius: 10px;">
                        <h4 style="color: {status_color}; margin: 0;">{status_msg}</h4>
                        <p style="color: white; font-size: 0.9rem; margin-top: 5px;">{advice}</p>
                    </div>
                """, unsafe_allow_html=True)

                # ميزة الربط بين الصفحات (نقل القيمة تلقائياً)
                st.write("---")
                if st.button("📝 استخدام هذه القيمة في التقييم الرئيسي", use_container_width=True):
                    st.session_state.home_glucose = glucose
                    st.balloons()
                    st.toast("تم تحديث قيمة الجلوكوز في صفحة التقييم!")
