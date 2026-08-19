import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

def video_stream(model, confidence_threshold):
    # กำหนดสถานะ run_cam ใน session_state หากยังไม่มี และบังคับให้เป็น True ทุกครั้งที่ฟังก์ชันถูกเรียก
    if "run_cam" not in st.session_state:
        st.session_state.run_cam = True
    else:
        st.session_state.run_cam = True 
        
    # เปิดการเชื่อมต่อกับกล้อง (0 คือกล้องตัวหลักติด Laptop, 1 2... คือกล้องตัวที่เชื่อมกับ USB) 
    cap = cv2.VideoCapture(0) 
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)     
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
    cap.set(cv2.CAP_PROP_FPS, 30) 
    
    if not cap.isOpened():
        st.error("ไม่สามารถเข้าถึงกล้องได้ โปรดตรวจสอบการเชื่อมต่อ")
        return

    frame_placeholder = st.empty()
    count_placeholder = st.empty()
    stop_placeholder = st.empty()

    # หากมีการกดปุ่ม Stop Webcam ให้เปลี่ยนสถานะ run_cam เป็น False
    if stop_placeholder.button("Stop Webcam", key="stop_btn"):
        st.session_state.run_cam = False

    # ใช้ try/finally เพื่อรับประกันว่ากล้องจะถูก release เสมอ
    try:
        # เพิ่ม st.session_state.run_cam เป็นเงื่อนไขในลูป while
        while cap.isOpened() and st.session_state.run_cam:
            ret, frame = cap.read()

            if not ret: 
                st.error("ไม่สามารถอ่าน Frame จากกล้องได้") 
                break

            results = model(frame, conf=confidence_threshold, stream=False) 
            annotated_frame = results[0].plot()
            person_count = len(results[0].boxes) if results[0].boxes is not None else 0   

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) 
            frame_placeholder.image(frame_rgb, channels="RGB", width=720) 
            count_placeholder.markdown(f"### 🏃 จำนวนคนที่ตรวจพบ: `{person_count}` คน")

    finally:
        # โค้ดส่วนนี้จะทำงานเสมอ ไม่ว่าจะหลุดจากลูปปกติ หรือถูก Streamlit ขัดจังหวะการทำงาน (Rerun)
        cap.release()
        cv2.destroyAllWindows() 

def main():
    st.set_page_config(page_title="Pose Estimation AI", layout="wide")  
    
    st.title("🏃 Real-time CCTV Pose Estimation")  
    st.sidebar.header("งานวิจัยของศุภกร วงษ์เรืองพิบูล")
    st.sidebar.write("ทดสอบประสิทธิภาพโมเดลประเภท Pose estimation ตรวจจับการเคลื่อนไหวมนุษย์")
    st.sidebar.write(" ")

    model_path = "yolo11s-pose.pt"
    
    try:
        model = YOLO(model_path) 
        st.sidebar.success("โหลดโมเดล Pose สำเร็จ") 
    except Exception as e: 
        st.sidebar.error(f"ไม่สามารถโหลดโมเดลได้: {e}") 
        return

    # Slider สำหรับปรับ Confidence
    confidence_threshold = st.sidebar.slider(
        "ค่าความเชื่อมั่น (Confidence Threshold)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,  
        step=0.05   
    )

    if st.sidebar.button("เปิดกล้อง (Start Webcam)"):   
        video_stream(model, confidence_threshold) 

if __name__ == "__main__":  
    main()
