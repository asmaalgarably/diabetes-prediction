import os
import pickle
import re
import tempfile
import warnings
import arabic_reshaper
import easyocr
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from bidi.algorithm import get_display
from fpdf import FPDF
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get(
    "\ufeffGEMINI_API_KEY")
genai.configure(api_key=api_key)
ai_model = genai.GenerativeModel('models/gemini-2.5-flash-lite')

 


# ------------------- 1.Comprehensive Linguistic Dictionary-------------------
languages = {
    "العربية": {
        "dir": "ltr",
        "nav_title": "لوحة التحكم",
        "nav_page": "اختر الصفحة",
        "home": "الصفحة الرئيسية",
        "risk": "تقييم خطر المريض",
        "report": "التقرير الطبي",
        "batch": "التحليل الجماعي",
        "image": "تحليل الصور (OCR)",
        "ai": "المساعد الطبي الذكي",
        "hero_title": "Smart Diabetes AI",
        "ai_title": "مستشار DiaVision للذكاء الاصطناعي",
        "hero_subtitle": "النظام الذكي المتكامل للوقاية من السكري",
        "hero_desc": "نجمع بين دقة البيانات وقوة الذكاء الاصطناعي لنقدم لك تحليلاً طبياً استباقياً يساعدك في اتخاذ قراراتك الصحية بثقة.",
        "f1_t": "تنبؤ دقيق", "f1_d": "تحليل المخاطر باستخدام خوارزميات متقدمة.",
        "f2_t": "تقارير شاملة", "f2_d": "توليد ملفات PDF تحتوي على كافة التفاصيل.",
        "f3_t": "قراءة الصور", "f3_d": "استخراج النتائج من صور الفحوصات الطبية.",
        "f4_t": "تحليل الدفعات", "f4_d": "معالجة بيانات مجموعة مرضى دفعة واحدة.",
        "form_h": "🧑‍⚕️ نموذج تقييم بيانات المريض",
        "form_p": "يرجى إدخال البيانات الحيوية بدقة لضمان دقة التنبؤ",
        "l_name": "اسم المريض بالكامل", "l_age": "العمر (سنة)", "l_gender": "الجنس",
        "l_glucose": "مستوى الجلوكوز (mg/dL)", "l_weight": "الوزن (كجم)", "l_height": "الطول (سم)",
        "l_hyper": "هل يعاني من ضغط الدم؟", "l_family": "تاريخ عائلي للسكري؟",
        "btn_save": "💾 حفظ البيانات وتحليلها",
        "rep_title": "🏥 لوحة تشخيص حالة المريض",
        "risk_level": "مستوى الخطورة", "prob": "احتمالية الإصابة",
        "advice_title": "📋 التوصيات الطبية المخصصة",
        "low": "منخفض", "med": "متوسط", "high": "مرتفع",
        "btn_pdf": "📥 تحميل التقرير الطبي الكامل (PDF)",
        "ocr_h": "🧪 المختبر الذكي | AI Image Lab",
        "ocr_btn": "📝 استخدام هذه القيمة في التقييم",
        "batch_h": "📊 تحليل بيانات المجموعة",
        "gender_m": "ذكر", "gender_f": "أنثى",
        "yes": "نعم", "no": "لا",
        "success": "تم تحديث البيانات بنجاح!" ,
        "info": "يمكنك الآن الانتقال لصفحة التقرير"
      
    },
    "English": {
        "dir": "ltr",
        "nav_title": "Control Panel",
        "nav_page": "Navigation",
        "home": "Home Page",
        "risk": "Risk Assessment",
        "report": "Medical Report",
        "batch": "Batch Analysis",
        "image": "Image Analysis (OCR)",
        "ai": "AI Medical Assistant",
        "ai_title": "DiaVision AI Consultant",
        "hero_title": "Smart Diabetes AI",
        "hero_subtitle": "Integrated Smart System for Diabetes Prevention",
        "hero_desc": "We combine data accuracy with AI power to provide proactive medical analysis.",
        "f1_t": "AI Prediction", "f1_d": "Risk analysis using advanced algorithms.",
        "f2_t": "Comprehensive Reports", "f2_d": "Generate PDF files with full details.",
        "f3_t": "Medical OCR", "f3_d": "Extract results from medical test images.",
        "f4_t": "Batch Processing", "f4_d": "Process multiple patient data at once.",
        "form_h": "🧑‍⚕️ Patient Assessment Form",
        "form_p": "Please enter vital data accurately",
        "l_name": "Full Patient Name", "l_age": "Age (Years)", "l_gender": "Gender",
        "l_glucose": "Glucose Level (mg/dL)", "l_weight": "Weight (kg)", "l_height": "Height (cm)",
        "l_hyper": "Hypertension?", "l_family": "Family History?",
        "btn_save": "💾 Save & Analyze",
        "rep_title": "🏥 Medical Dashboard",
        "risk_level": "Risk Level", "prob": "Probability",
        "advice_title": "📋 Custom Recommendations",
        "low": "Low", "med": "Medium", "high": "High",
        "btn_pdf": "📥 Download PDF Report",
        "ocr_h": "🧪 AI Image Lab",
        "ocr_btn": "📝 Apply to Assessment",
        "batch_h": "📊 Batch Data Analysis",
        "gender_m": "Male", "gender_f": "Female",
        "yes": "Yes", "no": "No",
          "success": "Data updated successfully!",
          "info": "You can now view the report",
    }
}

# ------------------- 2.Settings and download-------------------
st.set_page_config(page_title="Smart Diabetes AI", layout="wide", page_icon="💉")

# مسارات الملفات 
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "models", "rf_diabetes_model.pkl")
scaler_path = os.path.join(current_dir, "..", "models", "rf_scaler.pkl")
columns_path = os.path.join(current_dir, "..", "models", "rf_columns.pkl")


# Images
logo_path = os.path.join(current_dir, "..", "image", "logo.png")
logo_path1 = os.path.join(current_dir, "..", "image", "logo1.png")

# Fonts
font_path = os.path.join(current_dir, "Fonts", "DejaVuSans.ttf")

@st.cache_resource
def load_assets():
    with open(model_path, "rb") as f: m = pickle.load(f)
    with open(scaler_path, "rb") as f: s = pickle.load(f)
    with open(columns_path, "rb") as f: c = pickle.load(f)
    reader = easyocr.Reader(['ar', 'en'], gpu=False)
    return m, s, c, reader

try:
    model, scaler, model_columns, reader = load_assets()
except:
    st.error("⚠️ Model files not found!")
    st.stop()

# ------------------- 3. Side menu-------------------

lang_choice = st.sidebar.selectbox(
    "Language | اللغة", ["العربية", "English"], key="global_lang_sel")
t = languages[lang_choice]

if "nav_idx" not in st.session_state:
    st.session_state.nav_idx = 0

menu_options = [t["home"], t["risk"],
                t["report"], t["batch"], t["image"], t["ai"]]

page = st.sidebar.radio(
    t["nav_page"],
    menu_options,
    index=st.session_state.nav_idx
)

st.session_state.nav_idx = menu_options.index(page)
# ------------------- 4.(CSS) -------------------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    .stApp {{ background-color: #0b0e14; font-family: 'Cairo', sans-serif; direction: {t['dir']}; }}
    
    .hero-gradient {{
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.9) 0%, rgba(59, 130, 246, 0.8) 100%);
        padding: 60px 20px; border-radius: 30px; text-align: center; margin-bottom: 50px;
    }}

    .feature-card {{
        background: #161b22; 
        padding: 30px; 
        border-radius: 20px 20px 0 0; 
        border: 1px solid #30363d;
        border-top: 4px solid #3b82f6; 
        text-align: center; 
        transition: 0.4s;
        min-height: 280px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-bottom: 0px; 
    }}

    .stButton>button {{
    width: 100%;
        border-radius: 0 0 20px 20px !important; 
        /* تدرج لوني أزرق فخم */
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        height: 55px;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.4s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }}
    
    .stButton>button:hover {{
        / * تغيير التدرج وزيادة الإضاءة عند التمرير * /
    background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }}

    .stButton>button:active {{
    transform: translateY(1px) !important;
    }}

    .feature-card:hover {{ transform: translateY(-5px); border-color: #3b82f6; }}
    </style>
    """, unsafe_allow_html=True)

# generate_pdf
def generate_pdf(patient_info, weekly_plan):
    pdf = FPDF()
    pdf.add_page()

 
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    effective_width = pdf.w - 30  

    pdf.add_font("DejaVu", "", font_path)
    pdf.add_font("DejaVu", "B", font_path)

    def fix_text(text):
        if not text:
            return ""
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)

    # --- 1.  (Logo) ---
    if os.path.exists(logo_path1):
        
        pdf.image(logo_path1, x=(pdf.w - 40) / 2, y=10, w=40)
        pdf.ln(35)
    else:
        pdf.ln(10)

    # --- 2. Main Title---
    pdf.set_font("DejaVu", "B", 18)
    title = "Medical Report | تقرير طبي"
    pdf.cell(effective_width, 10, fix_text(title), 0, 1, 'C')
    pdf.ln(5)

    # --- 3. Basic Patient Information ---
    pdf.set_font("DejaVu", "", 12)
    for key, val in patient_info.items():
        if key != "Advice":
            label = fix_text(f"{key}:")
            value = fix_text(val)
           
            pdf.cell(effective_width * 0.3, 10, label, 1, 0,
                     'R' if lang_choice == "العربية" else 'L')
            pdf.cell(effective_width * 0.7, 10, value, 1, 1,
                     'R' if lang_choice == "العربية" else 'L')

    # --- 4. Proposed weekly schedule---
    pdf.ln(10)
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(effective_width, 10, fix_text("Weekly Plan | الخطة الأسبوعية"),
             0, 1, 'R' if lang_choice == "العربية" else 'L')

    pdf.set_font("DejaVu", "", 10)
    pdf.set_fill_color(240, 240, 240)

    day_w = effective_width * 0.25
    plan_w = effective_width * 0.75

    if lang_choice == "العربية":
        pdf.cell(plan_w, 10, fix_text("النشاط"), 1, 0, 'C', True)
        pdf.cell(day_w, 10, fix_text("اليوم"), 1, 1, 'C', True)
        for day, plan in weekly_plan.items():
            pdf.cell(plan_w, 10, fix_text(plan), 1, 0, 'R')
            pdf.cell(day_w, 10, fix_text(day), 1, 1, 'R')
    else:
        pdf.cell(day_w, 10, fix_text("Day"), 1, 0, 'C', True)
        pdf.cell(plan_w, 10, fix_text("Activity"), 1, 1, 'C', True)
        for day, plan in weekly_plan.items():
            pdf.cell(day_w, 10, fix_text(day), 1, 0, 'L')
            pdf.cell(plan_w, 10, fix_text(plan), 1, 1, 'L')

    # --- 5.General advice  ---
 
    pdf.ln(10)
    pdf.set_font("DejaVu", "B", 14)
    
    pdf.cell(effective_width, 10, fix_text(
        t['advice_title']), 0, 1, 'R' if lang_choice == "العربية" else 'L')

    pdf.set_font("DejaVu", "", 11)

    advice_data = patient_info.get("Advice", [])

 
    for item in advice_data:
       
        clean_item = fix_text(item)
 
        pdf.multi_cell(effective_width, 8, clean_item, border=0,
                       align='R' if lang_choice == "العربية" else 'L')

       
        pdf.ln(2)

    return bytes(pdf.output())
# ------------------- 6. Pages -------------------


if page == t["home"]:
  
    if os.path.exists(logo_path):
        st.columns([1, 1, 1])[1].image(logo_path, width='stretch')

    st.markdown(f"""
        <div class="hero-gradient">
            <h1 style="font-size: 3.5rem; color: white; margin:0;">{t['hero_title']}</h1>
            <p style="font-size: 1.4rem; color: #e6edf3; margin-top:10px;">{t['hero_subtitle']}</p>
        </div>
    """, unsafe_allow_html=True)

  
    feats = [
        ("🧬", t["f1_t"], t["f1_d"], 1),  # 1 (Risk)
        ("📋", t["f2_t"], t["f2_d"], 2),  # 2 (Report)
        ("🔍", t["f3_t"], t["f3_d"], 4),  # 4 (OCR)
        ("📊", t["f4_t"], t["f4_d"], 3),  # 3 (Batch)
        ("🩺", t["ai_title"], "Assistant", 5) # 5 (AI)
    ]

    c1, c2, c3 = st.columns(3)
    row1_cols = [c1, c2, c3]
    for i in range(3):
        with row1_cols[i]:
            st.markdown(f"""<div class="feature-card"><h3>{feats[i][0]}</h3>
                        <h4 style="color:#58a6ff;">{feats[i][1]}</h4>
                        <p style="color:#8b949e; font-size:0.85rem;">{feats[i][2]}</p></div>""", unsafe_allow_html=True)

           
            if st.button(f"{'دخول' if lang_choice == 'العربية' else 'Enter'}", key=f"btn_nav_{i}"):
                st.session_state.nav_idx = feats[i][3]  
                st.rerun()  

    st.markdown("<br>", unsafe_allow_html=True)
 
    _, c4, c5, _ = st.columns([0.5, 1, 1, 0.5])
    row2_cols = [c4, c5]
    for i in range(3, 5):
        with row2_cols[i-3]:
            st.markdown(f"""<div class="feature-card"><h3>{feats[i][0]}</h3>
                        <h4 style="color:#58a6ff;">{feats[i][1]}</h4>
                        <p style="color:#8b949e; font-size:0.85rem;">{feats[i][2]}</p></div>""", unsafe_allow_html=True)

            if st.button(f"{'دخول' if lang_choice == 'العربية' else 'Enter'}", key=f"btn_nav_{i}"):
                st.session_state.nav_idx = feats[i][3]   
                st.rerun()
            
# =================== PAGE 2: PATIENT RISK ASSESSMENT ===================
elif page == t["risk"]:
    def handle_transition():
        st.session_state.page = t["report"]
        st.session_state.form_submitted = False  

    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 25px;">
            <h2 style="color: #3b82f6; font-family: 'Cairo', sans-serif;">{t['form_h']}</h2>
            <p style="color: #8b949e;">{t['form_p']}</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form_modern"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(t["l_name"], placeholder="John Doe")
            age = st.number_input(
                t["l_age"], min_value=1, max_value=120, value=30)
            gender = st.selectbox(
                t["l_gender"], [t["gender_m"], t["gender_f"]])
             
            if "home_glucose" not in st.session_state:
                st.session_state["home_glucose"] = 100.0

            
            glucose = st.number_input(
                t["l_glucose"],
                min_value=40.0,
                max_value=500.0,
                value=float(st.session_state["home_glucose"]), 
                key="risk_glucose_master" 
)

        with col2:
            weight = st.number_input(
                t["l_weight"], min_value=10.0, max_value=250.0, value=70.0)
            height = st.number_input(
                t["l_height"], min_value=50.0, max_value=250.0, value=170.0)
            hyper = st.selectbox(t["l_hyper"], [t["no"], t["yes"]])
            family = st.selectbox(t["l_family"], [t["no"], t["yes"]])

       
        submit_button = st.form_submit_button(
            t["btn_save"], width='stretch')

  
    if submit_button:
        if name.strip() == "":
            st.error("⚠️ " + ("الرجاء إدخال الاسم" if lang_choice ==
                     "العربية" else "Please enter name"))
        else:
            st.session_state.update({
                "home_name": name,
                "home_age": age,
                "home_gender": gender,
                "home_glucose": glucose,
                "home_weight": weight,
                "home_height": height,
                "home_hypertension": hyper,
                "home_family_diabetes": family,
                "form_submitted": True  
            })
            st.success(f"✅ {name} - {t['success']}")

   
    if st.session_state.get("form_submitted"):
        st.markdown("<br>", unsafe_allow_html=True)
 
        if st.button(f"➡️ {t['report']}", width='stretch', type="primary"):
        
            st.session_state.nav_idx = 2
       
            st.rerun()

# =================== PAGE 3: MEDICAL REPORT ===================
elif page == t["report"]:
    st.markdown(
        f"<h2 style='text-align: center; color: #58a6ff; font-family: \"Cairo\", sans-serif;'>{t['rep_title']}</h2>", unsafe_allow_html=True)

    col_input, col_result = st.columns([0.8, 2], gap="large")

    with col_input:
        st.markdown(f"#### ⚙️ {t['nav_title']}")
        with st.form("medical_report_form"):
            
            name = st.session_state.get("home_name", "---")
            age = st.session_state.get("home_age", 0)
            glucose = st.session_state.get("home_glucose", 0)
            weight = st.session_state.get("home_weight", 0)
            height = st.session_state.get("home_height", 1)
            gender = st.session_state.get("home_gender", t["gender_m"])
            hypertensive = st.session_state.get("home_hypertension", t["no"])
            family_diabetes = st.session_state.get(
                "home_family_diabetes", t["no"])

            st.write(f"👤 **{t['l_name']}:** {name}")
            st.write(f"🩸 **{t['l_glucose']}:** {glucose} mg/dL")

            submit = st.form_submit_button(
                "📊 " + ("تحليل النتيجة" if lang_choice == "العربية" else "Analyze Result"), width='stretch')

    with col_result:
        if not submit:
            st.markdown(f"""
                <div style="text-align: center; padding: 100px 20px; border: 2px dashed #30363d; border-radius: 30px; background: #0d1117;">
                    <h3 style="color: #8b949e;">{'بانتظار بدء التحليل...' if lang_choice == 'العربية' else 'Waiting for analysis...'}</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
           
            bmi = round(weight / ((height / 100) ** 2), 2)
            gen_val = 1 if gender in ["Male", "ذكر"] else 0
            fam_val = 1 if family_diabetes in ["Yes", "نعم"] else 0
            hyp_val = 1 if hypertensive in ["Yes", "نعم"] else 0

            new_data = pd.DataFrame({'age': [age], 'gender': [gen_val], 'bmi': [bmi], 'glucose': [
                                    glucose], 'family_diabetes': [fam_val], 'hypertensive': [hyp_val]})
            for col in model_columns:
                if col not in new_data.columns:
                    new_data[col] = 0

            prob = model.predict_proba(
                scaler.transform(new_data[model_columns]))[0][1]
 
            if prob < 0.33:
                risk_txt, color, icon = t["low"], "#2ecc71", "🟢"
                if lang_choice == "العربية":
                    advice_list = [
                        "• حافظ على وزن مثالي", "• فحص دوري كل 6 أشهر", "• تقليل السكريات المضافة"]
                    weekly_plan = {"السبت-الأحد": "مشي 30 دقيقة", "الاثنين-الثلاثاء": "نظام قليل السكر",
                                   "الأربعاء-الخميس": "تمارين مرونة", "الجمعة": "يوم راحة"}
                else:
                    advice_list = ["• Maintain healthy weight",
                                   "• Checkup every 6 months", "• Reduce added sugars"]
                    weekly_plan = {"Sat-Sun": "30 min Walking", "Mon-Tue": "Low Sugar Diet",
                                   "Wed-Thu": "Flexibility Training", "Friday": "Rest Day"}

            elif prob < 0.66:
                risk_txt, color, icon = t["med"], "#f39c12", "🟡"
                if lang_choice == "العربية":
                    advice_list = ["• قلل النشويات والخبز الأبيض",
                                   "• ممارسة الرياضة 5 أيام أسبوعياً", "• شرب الماء بكثرة"]
                    weekly_plan = {"السبت-الأحد": "مشي سريع 45 د", "الاثنين-الثلاثاء": "صيام متقطع 14 ساعة",
                                   "الأربعاء-الخميس": "تمارين كارديو", "الجمعة": "فحص السكر"}
                else:
                    advice_list = ["• Reduce carbs & white bread",
                                   "• Exercise 5 days/week", "• Drink plenty of water"]
                    weekly_plan = {"Sat-Sun": "45 min Brisk Walk", "Mon-Tue": "14h Fasting",
                                   "Wed-Thu": "Cardio Session", "Friday": "Glucose Check"}

            else:
                risk_txt, color, icon = t["high"], "#e74c3c", "🔴"
                if lang_choice == "العربية":
                    advice_list = [
                        "• استشر طبيباً فوراً", "• حمية غذائية صارمة (كيتو أو لو كارب)", "• فحص السكر اليومي صائم وفاطر"]
                    weekly_plan = {"السبت-الأحد": "مشي خفيف + أدوية", "الاثنين-الثلاثاء": "حذف السكر تماماً",
                                   "الأربعاء-الخميس": "زيارة الطبيب", "الجمعة": "فحص دم شامل"}
                else:
                    advice_list = ["• Consult doctor immediately",
                                   "• Strict Low-Carb Diet", "• Daily Glucose Monitoring"]
                    weekly_plan = {"Sat-Sun": "Light Walk + Meds", "Mon-Tue": "Zero Sugar Plan",
                                   "Wed-Thu": "Specialist Visit", "Friday": "Full Blood Lab"}

          
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="metric-box" style="border-top: 4px solid {color};"><h4>{t["risk_level"]}</h4><h2 style="color:{color}">{icon} {risk_txt}</h2></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div class="metric-box"><h4>{t["prob"]}</h4><h2>{prob*100:.1f}%</h2></div>', unsafe_allow_html=True)

       
            st.markdown(
                f"### 📅 {'الخطة الأسبوعية المقترحة' if lang_choice == 'العربية' else 'Weekly Plan'}")
            st.table(pd.DataFrame(weekly_plan.items(), columns=[
                     "Day" if lang_choice == "English" else "اليوم", "Plan" if lang_choice == "English" else "الخطة"]))
 
            st.markdown(f"### {t['advice_title']}")
            for item in advice_list:
                st.write(item)
 
            pdf_data = {
                "Name": name, "Age": age, "Gender": gender, "BMI": bmi,
                "Glucose": glucose, "Risk Level": f"{risk_txt} ({prob*100:.1f}%)",
                "Advice": advice_list  
            }

            pdf_bytes = generate_pdf(pdf_data, weekly_plan)
            st.download_button(t["btn_pdf"], pdf_bytes,
                               f"Report_{name}.pdf", width='stretch')
# =================== PAGE: BATCH ANALYSIS ===================
elif page == t["batch"]:
    title_text = "📊 التحليل الجماعي والبيانات الضخمة" if lang_choice == "العربية" else "📊 Batch Analysis & Big Data"
    st.markdown(
        f"<h2 style='text-align: center;'>{title_text}</h2>", unsafe_allow_html=True)

    upload_msg = (
        "يرجى رفع ملف Excel أو CSV يحتوي على بيانات المرضى (name, age, height, weight, glucose, gender, family_diabetes, hypertensive)"
        if lang_choice == "العربية" else
        "Please upload an Excel or CSV file containing patient data (name, age, height, weight, glucose, gender, family_diabetes, hypertensive)"
    )

    st.markdown(f"""
        <div style="background: #161b22; padding: 20px; border-radius: 15px; border: 1px dashed #3b82f6; text-align: center; margin-bottom: 25px;">
            <p style="margin-bottom: 10px; color: #8b949e;">{upload_msg}</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

    if uploaded_file:
        
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(
            ".csv") else pd.read_excel(uploaded_file)

        success_msg = f"✅ تم تحميل الملف بنجاح! يحتوي على {len(df)} مريض." if lang_choice == "العربية" else f"✅ File uploaded! Contains {len(df)} patients."
        st.success(success_msg)

        results = []
        
        for _, row in df.iterrows():
            
            bmi = round(row['weight'] / ((row['height'] / 100) ** 2), 2)

            new_data = pd.DataFrame({
                'age': [row['age']],
                'gender': [1 if str(row['gender']).lower() in ["male", "ذكر"] else 0],
                'bmi': [bmi],
                'glucose': [row['glucose']],
                'family_diabetes': [1 if str(row['family_diabetes']).lower() in ["yes", "نعم"] else 0],
                'hypertensive': [1 if str(row['hypertensive']).lower() in ["yes", "نعم"] else 0]
            })

            for col in model_columns:
                if col not in new_data.columns:
                    new_data[col] = 0

            prob = model.predict_proba(
                scaler.transform(new_data[model_columns]))[0][1]

           
            if prob > 0.66:
                risk_status = "High" if lang_choice == "English" else "عالية"
                status_key = "High" 
            elif prob > 0.33:
                risk_status = "Medium" if lang_choice == "English" else "متوسطة"
                status_key = "Medium"
            else:
                risk_status = "Low" if lang_choice == "English" else "منخفضة"
                status_key = "Low"

            results.append({
                "Name" if lang_choice == "English" else "اسم المريض": row['name'],
                "Glucose" if lang_choice == "English" else "الجلوكوز": row['glucose'],
                "BMI": bmi,
                "Probability" if lang_choice == "English" else "الاحتمالية": f"{prob*100:.1f}%",
                "Status" if lang_choice == "English" else "الحالة": risk_status,
                "Internal_Status": status_key   
            })

        results_df = pd.DataFrame(results)

        st.markdown(
            f"### 📈 {'ملخص الدفعة' if lang_choice == 'العربية' else 'Batch Summary'}")
        k1, k2, k3, k4 = st.columns(4)

        total_patients = len(results_df)
        
        high_risk_count = len(
            results_df[results_df['Internal_Status'] == "High"])
        avg_glucose = round(
            results_df["Glucose" if lang_choice == "English" else "الجلوكوز"].mean(), 1)

        labels = {
            "total": "إجمالي المرضى" if lang_choice == "العربية" else "Total Patients",
            "high": "مخاطر عالية" if lang_choice == "العربية" else "High Risk",
            "avg": "متوسط الجلوكوز" if lang_choice == "العربية" else "Avg Glucose",
            "ratio": "نسبة الخطورة" if lang_choice == "العربية" else "Risk Ratio"
        }

        k1.markdown(
            f'<div class="metric-box-v2"><h5>{labels["total"]}</h5><h2 style="color:#58a6ff;">{total_patients}</h2></div>', unsafe_allow_html=True)
        k2.markdown(
            f'<div class="metric-box-v2"><h5>{labels["high"]}</h5><h2 style="color:#e74c3c;">{high_risk_count}</h2></div>', unsafe_allow_html=True)
        k3.markdown(
            f'<div class="metric-box-v2"><h5>{labels["avg"]}</h5><h2 style="color:#f39c12;">{avg_glucose}</h2></div>', unsafe_allow_html=True)
        k4.markdown(
            f'<div class="metric-box-v2"><h5>{labels["ratio"]}</h5><h2 style="color:#2ecc71;">{round((high_risk_count/total_patients)*100)}%</h2></div>', unsafe_allow_html=True)

        st.write("---")

 
        col_tab, col_chart = st.columns([1.5, 1])

        with col_tab:
            st.markdown(
                f"#### 📋 {'تفاصيل المرضى' if lang_choice == 'العربية' else 'Patient Details'}")

            def color_risk(val):
                
                if val in ["High", "عالية"]:
                    color = '#e74c3c'
                elif val in ["Medium", "متوسطة"]:
                    color = '#f39c12'
                else:
                    color = '#2ecc71'
                return f'color: {color}; font-weight: bold'

            
            display_df = results_df.drop(columns=['Internal_Status'])
            st.dataframe(display_df.style.applymap(color_risk, subset=[
                         display_df.columns[-1]]), width='stretch')

        with col_chart:
            st.markdown(
                f"#### 📊 {'توزيع المخاطر' if lang_choice == 'العربية' else 'Risk Distribution'}")
            status_counts = results_df['Internal_Status'].value_counts()

            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#0d1117')

            colors_map = {'High': '#e74c3c',
                          'Medium': '#f39c12', 'Low': '#2ecc71'}
            ordered_colors = [colors_map.get(
                x, '#58a6ff') for x in status_counts.index]

            status_counts.plot(kind='pie', autopct='%1.1f%%',
                               colors=ordered_colors, ax=ax, textprops={'color': "w"})
            ax.set_ylabel('')
            st.pyplot(fig)

     
        st.write("---")
        csv_results = results_df.to_csv(index=False).encode('utf-8-sig')
        btn_label = "📥 تحميل نتائج التحليل بالكامل (CSV)" if lang_choice == "العربية" else "📥 Download All Results (CSV)"
        st.download_button(label=btn_label, data=csv_results,
                           file_name="Batch_Analysis.csv", mime="text/csv", width='stretch')
 # =================== PAGE: MEDICAL IMAGE ANALYSIS (COMPLETE & CLEAN) ===================
elif page == t["image"]:
    if "home_glucose" not in st.session_state:
        st.session_state["home_glucose"] = 100.0

    title_text = "🧪 المختبر الذكي | AI Image Lab" if lang_choice == "العربية" else "🧪 Smart Lab | AI Image Lab"
    subtitle_text = (
        "تحليل صور الفحوصات الطبية واستخراج النتائج فوراً"
        if lang_choice == "العربية" else
        "Extract medical test results from images instantly"
    )

    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #58a6ff; font-family: 'Cairo', sans-serif;">{title_text}</h1>
            <p style="color: #8b949e; font-size: 1.1rem;">{subtitle_text}</p>
        </div>
    """, unsafe_allow_html=True)
 
    upload_msg = "⬇️ ارفع صورة الفحص الطبية هنا (JPG, PNG, JPEG)" if lang_choice == "العربية" else "⬇️ Upload lab report image (JPG, PNG, JPEG)"
    st.markdown(
        f'<p style="text-align: {"right" if lang_choice == "العربية" else "left"}; color: #58a6ff;">{upload_msg}</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        st.markdown("<br>", unsafe_allow_html=True)
        col_img, spacer, col_res = st.columns([1, 0.1, 1])

        with col_img:
            st.markdown(
                f"### 🖼️ {'الصورة المرفوعة' if lang_choice == 'العربية' else 'Uploaded Scan'}")
            image = Image.open(uploaded_file)
            st.markdown('<div style="border: 2px solid #30363d; border-radius: 15px; padding: 10px; background: #0d1117; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">', unsafe_allow_html=True)
            st.image(image, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            st.markdown(
                f"### 📋 {'نتائج التحليل' if lang_choice == 'العربية' else 'Analysis Result'}")

            with st.status("🔍 Scanning..." if lang_choice == "English" else "🔍 جاري الفحص...", expanded=True) as status:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    image.save(tmp.name)
                    img_path = tmp.name

                results = reader.readtext(img_path)
                full_text = " ".join([r[1] for r in results])

                glucose = None
                patterns = [
                    r'Glucose[:\s]*([0-9]{2,3})', r'([0-9]{2,3})\s*mg', r'سكر[:\s]*([0-9]{2,3})']
                for p in patterns:
                    match = re.search(p, full_text, re.IGNORECASE)
                    if match:
                        glucose = float(match.group(1))
                        break

                status.update(label="✅ Success!" if lang_choice ==
                              "English" else "✅ تم استخراج البيانات", state="complete")

            if glucose:
                 
                label_glucose = "مستوى الجلوكوز" if lang_choice == "العربية" else "Glucose Level"
                st.markdown(f"""
                    <div style="background: #161b22; padding: 25px; border-radius: 20px; border-top: 5px solid #3b82f6; text-align: center; margin-bottom: 20px;">
                        <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 5px;">{label_glucose}</p>
                        <h1 style="color: #ffffff; font-size: 4.5rem; margin: 0; font-family: 'Courier New';">{int(glucose)}</h1>
                        <p style="color: #3b82f6; font-size: 1.2rem; font-weight: bold;">mg/dL</p>
                    </div>
                """, unsafe_allow_html=True)

                if glucose < 70:
                    status_msg, status_color, icon = (
                        "Low" if lang_choice == "English" else "منخفض", "#f1c40f", "⚠️")
                    advice = "Need sugar source." if lang_choice == "English" else "تحتاج لمصدر سكر سريع."
                elif 70 <= glucose <= 140:
                    status_msg, status_color, icon = (
                        "Normal" if lang_choice == "English" else "طبيعي", "#2ecc71", "✅")
                    advice = "Healthy range." if lang_choice == "English" else "ضمن النطاق الصحي."
                else:
                    status_msg, status_color, icon = (
                        "High" if lang_choice == "English" else "مرتفع", "#e74c3c", "🚨")
                    advice = "Consult a doctor." if lang_choice == "English" else "يرجى استشارة طبيب."

                st.markdown(f"""
                    <div style="display: flex; align-items: center; background: {status_color}15; padding: 15px; border-radius: 12px; border: 1px solid {status_color}40;">
                        <div style="font-size: 2rem; margin-right: 15px; margin-left: {'0' if lang_choice == 'English' else '15px'};">{icon}</div>
                        <div style="flex-grow: 1; text-align: {'left' if lang_choice == 'English' else 'right'};">
                            <h4 style="color: {status_color}; margin: 0; font-size: 1.2rem;">{status_msg}</h4>
                            <p style="color: white; font-size: 0.85rem; margin-top: 3px;">{advice}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

        
                btn_sync = "🚀 مزامنة مع التقييم الطبي" if lang_choice == "العربية" else "🚀 Sync to Medical Assessment"
                if st.button(btn_sync,width='stretch', type="primary"):
                                    
                    st.session_state["home_glucose"] = glucose
                    st.session_state["risk_glucose_master"] = float(glucose)
                    st.session_state["page"] = t["risk"]

                    st.rerun()
            else:
                st.error("❌ Could not read glucose. | لم نتمكن من قراءة القيمة.")
         
    with st.expander("ℹ️ " + ("كيفية الاستخدام" if lang_choice == "العربية" else "How to Use")):
        if lang_choice == "العربية":
            st.write("1. ارفع صورة واضحة لتقرير المختبر.")
            st.write("2. سيقوم النظام بقراءة النصوص آلياً.")
            st.write("3. اضغط على 'مزامنة' لنقل القيم تلقائياً.")
        else:
            st.write("1. Upload a clear lab report image.")
            st.write("2. The system scans text automatically.")
            st.write("3. Click 'Sync' to transfer data instantly.")


# =================== PAGE: AI MEDICAL CONSULTANT ===================
elif "المساعد الطبي الذكي" in page or "AI Medical Assistant" in page:
    display_title = t.get('ai_title', 'DiaVision AI Consultant')
    display_subtitle = 'مساعدك الطبي الذكي - اسأل بأي لغة' if lang_choice == 'العربية' else 'Your AI Medical Assistant - Ask in any language'

    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 30px; padding: 20px; background: rgba(88, 166, 255, 0.1); border-radius: 20px; border: 1px solid #58a6ff33;">
            <h1 style="color: #58a6ff; font-family: 'Cairo', sans-serif;">🩺 {display_title}</h1>
            <p style="color: #8b949e; font-size: 1.1rem;">{display_subtitle}</p>
        </div>
    """, unsafe_allow_html=True)
 
    if "medical_chat_history" not in st.session_state:
        st.session_state.medical_chat_history = []

    for message in st.session_state.medical_chat_history:
        avatar_icon = "👤" if message["role"] == "user" else "🩺"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    chat_placeholder = "اسأل عن الأعراض، الأدوية، أو التحاليل..." if lang_choice == "العربية" else "Ask about symptoms, meds, or labs..."
    user_input = st.chat_input(chat_placeholder)

    if user_input:
        st.session_state.medical_chat_history.append(
            {"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("🔍 جاري التحليل..." if lang_choice == "العربية" else "🔍 Analyzing..."):
                 
                glucose_val = st.session_state.get('home_glucose', None)
                if glucose_val:
                    context_text = f"The user's blood glucose level is {glucose_val} mg/dL. Analyze this specific value."
                else:
                    context_text = "No glucose value is available yet. Do not mention any numbers, just ask the user if they want to upload a lab report or ask a general question."
                instruction = """
                        You are "DiaVision AI", a professional medical consultant specializing in Diabetes but knowledgeable in general medicine. 

                        CORE CAPABILITIES:
                        1. AUTO-ANALYSIS: Analyze blood sugar values immediately when detected in context or chat.
                        2. CONVERSATIONAL MEMORY: Remember previous readings and questions to avoid repetition.
                        3. MEDICAL SCOPE: You can discuss any MEDICAL or HEALTH topics (like heart disease, blood pressure, general symptoms). 
                        4. STRICT NON-MEDICAL GUARDRAILS: Politely decline ONLY non-medical questions (like sports, politics, or universities).
                        5. DYNAMIC LANGUAGE: Respond in the same language the user uses.

                        TONE: Professional and supportive.
                        DISCLAIMER: Always mention that this is for informational purposes and they must see a doctor.
                        """

                try:
                  
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.medical_chat_history[-5:]])
                    full_prompt = f"""
                    {instruction}
                    
                    CONTEXT: {context_text}
                    
                    CHAT HISTORY:
                    {history_text}
                    
                    NEW USER QUESTION: {user_input}
                    """
                    response = ai_model.generate_content(full_prompt)
                    if response and response.text:
                        ai_reply = response.text
                        st.markdown(ai_reply)
                    
                        st.session_state.medical_chat_history.append(
                            {"role": "assistant", "content": ai_reply})
                    else:
                        st.error("⚠️ لم يتمكن الذكاء الاصطناعي من توليد رد.")

                except Exception as e:
                    st.error(f"⚠️ خطأ في الاتصال: {str(e)}")

    if st.sidebar.button("🗑️ " + ("مسح المحادثة" if lang_choice == "العربية" else "Clear Chat")):
        st.session_state.medical_chat_history = []
        st.rerun()

