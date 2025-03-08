import streamlit as st
import numpy as np
import joblib

# โหลดโมเดลและออบเจ็กต์ที่บันทึกไว้
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly.pkl")
selector = joblib.load("selector.pkl")
model = joblib.load("random_forest_model.pkl")

# ตั้งค่า CSS เพื่อให้เนื้อหามาอยู่ตรงกลาง
st.markdown(
    """
    <style>
        /* จัดให้รูปภาพอยู่ตรงกลาง */
        .image-container {
            display: flex;
            justify-content: center;
        }
        
        /* ปรับขนาดช่องกรอกข้อมูล */
        div.stButton > button {
            width: 100%;
            font-size: 18px;
        }
        
        /* ปรับความกว้างของ Sidebar */
        section[data-testid="stSidebar"] {
            width: 30% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# แสดงรูปภาพตรงกลาง
st.image("Dtbezn3nNUxytg04ajX7dx26Klc68gw7qdv5slNuqTasWJ.webp", width=600)  # กำหนดขนาดได้


# ตั้งค่าชื่อแอป
st.title("โปรแกรมเว็บทำนายโรคเบาหวาน")

# ส่วนกรอกข้อมูลให้อยู่ตรงกลาง
with st.container():
    st.header("📌 กรุณากรอกข้อมูลเบื้องต้น")

    # เลือกเพศ
    gender = st.radio("เพศ", ["ชาย", "หญิง"], index=0)

    age = st.number_input("อายุ (ปี)", min_value=0, max_value=120)
    glucose = st.number_input("ระดับน้ำตาลในเลือด (mg/dL)", min_value=0, max_value=200)
    blood_pressure = st.number_input("ความดันโลหิต (mmHg)", min_value=0, max_value=150)
    skin_thickness = st.number_input("ความหนาของผิวหนัง (mm)", min_value=0, max_value=100)
    insulin = st.number_input("ระดับอินซูลิน (IU/mL)", min_value=0, max_value=900)
    bmi = st.number_input("ค่าดัชนีมวลกาย (BMI)", min_value=0.0, max_value=70.0)
     # ถ้าเป็นเพศหญิง ให้กรอกจำนวนครั้งที่ตั้งครรภ์
    diabetes_pedigree = st.number_input("ประวัติโรคเบาหวานในครอบครัว (ถ้าไม่ทราบสามารถใส่ 0 ได้)", min_value=0.0, max_value=2.5)
    pregnancies = (
        st.number_input("คุณตั้งครรภ์มาแล้วกี่ครั้ง", min_value=0, max_value=20)
        if gender == "หญิง"
        else 0
    )
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

# ปุ่มกดพยากรณ์
if st.button("🔍 ทำนาย", use_container_width=True):
    # ทำการประมวลผลข้อมูล
    input_scaled = scaler.transform(input_data)
    input_poly = poly.transform(input_scaled)
    input_selected = selector.transform(input_poly)

    # ทำการพยากรณ์
    prediction = model.predict(input_selected)[0]
    prediction_proba = model.predict_proba(input_selected)[0][1]

    # แสดงผลลัพธ์
    st.subheader("🎯 ผลลัพธ์การพยากรณ์")
    if prediction == 1:
        st.error(f"🔴 มีแนวโน้มเป็นเบาหวาน (ความมั่นใจ: {prediction_proba:.2f})")
    else:
        st.success(f"🟢 ไม่มีแนวโน้มเป็นเบาหวาน (ความมั่นใจ: {prediction_proba:.2f})")
