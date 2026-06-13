import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import gradio as gr
from PIL import Image
import numpy as np
import os
from torchvision.models import mobilenet_v3_large
from rembg import remove

# --- Configuration and Constants ---
IMAGE_SIZE = 224
CLASS_NAMES = ['FaLan', 'Keaw_Sawoey', 'NamDokMai_Seethong', 'None']
NUM_CLASSES = len(CLASS_NAMES)

MODEL_SAVE_PATH = '1mobilenet_v3_mango_leaf_classifier.pth' 
device = torch.device('cpu')

# --- Transformations ---
val_test_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- Model Definition and Loading ---
def load_model(model_path, num_classes, device):
    model = mobilenet_v3_large(weights=None) 
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully")
    return model

model = load_model(MODEL_SAVE_PATH, NUM_CLASSES, device)
model.to(device)
model.eval()

# --- Prediction Function for Gradio ---
def predict_image(image: Image.Image):
    # 🌟 ดักตรงนี้ ห้ามส่งสตริงว่างเปล่า ให้ส่งค่าข้อความแนะนำไปแทนตัวแปรโล่ง ๆ
    if image is None:
        return {"กรุณาอัปโหลดรูปภาพ": 0.0}, "👋 **สถานะ:** รอการอัปโหลดรูปภาพใบมะม่วง..."

    # 1. ลบพื้นหลังด้วย rembg
    nobg_image = remove(image)
    image_np = np.array(nobg_image)
    
    # 2. บังคับถมสีเทากลาง [128, 128, 128] ตรงส่วนที่โปร่งแสงด้วย NumPy
    if image_np.shape[-1] == 4:
        rgb = image_np[:, :, :3]
        alpha = image_np[:, :, 3]
        rgb[alpha == 0] = [128, 128, 128]
        image_np = rgb

    # 3. ย่อขนาดภาพให้ด้านยาวที่สุดเท่ากับ 224 ด้วย OpenCV
    h, w = image_np.shape[:2]
    scale = IMAGE_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 4. ใช้ OpenCV ทำ Padding ขอบสีเทา 128 รอบทิศทางให้ได้ขนาดมิติ 224x224 เป๊ะ ๆ
    pad_h = IMAGE_SIZE - new_h
    pad_w = IMAGE_SIZE - new_w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    
    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[128, 128, 128])

    # 5. ส่งภาพเข้าท่อประมวลผล
    augmented = val_test_transforms(image=padded_image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

    # สร้าง Dictionary ผลลัพธ์
    predictions = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    # 6. หาคลาสที่โมเดลทายออกมาสูงที่สุด
    highest_class_idx = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[highest_class_idx]
    
    # 🌟 ปรับปรุงเงื่อนไขสตริงตรงนี้ ไม่ให้หน้าจอค้าง แคชหลุด
    if predicted_class in ['FaLan', 'Keaw_Sawoey', 'NamDokMai_Seethong']:
        notice_text = "💡 **คำแนะนำเพิ่มเติม:** ระบบวิเคราะห์ว่าเป็นใบมะม่วงในระบบ เนื่องจากโปรแกรมอยู่ในช่วงทดลองพัฒนา เพื่อความถูกต้องแม่นยำมากขึ้น ควรปรึกษาหรือสอบถามผู้เชี่ยวชาญ/ผู้รู้ร่วม"
    else:
        notice_text = "ℹ️ **หมายเหตุ:** ระบบวิเคราะห์ว่าภาพนี้ไม่ใช่ใบมะม่วง 3 สายพันธุ์หลักในระบบ หรือภาพไม่มีความชัดเจนเพียงพอ"

    return predictions, notice_text

# --- Gradio Interface Setup ---
APP_DESCRIPTION = """
###  ระบบจำแนกสายพันธุ์ใบมะม่วงด้วย Deep Learning (MobileNetV3)

**สายพันธุ์ที่รองรับ:** ฟ้าลั่น (FaLan), เขียวเสวย (Keaw_Sawoey), น้ำดอกไม้สีทอง (NamDokMai_Seethong)

---

**⚠️ Disclaimer (คำเตือน):** ระบบนี้สร้างขึ้นเพื่อทดลองจำแนกสายพันธุ์ใบมะม่วงเท่านั้น เนื่องจากโปรแกรมนี้ยังอยู่ในช่วงการพัฒนาเพื่อนำไปใช้งานจริง ตัวโมเดล AI จึงอาจมีความผิดพลาดในการประมวลผลได้ ผู้ใช้งานควรใช้วิจารณญาณในการพิจารณาคำตอบ และแนะนำให้ใช้ภาพถ่ายใบมะม่วงแบบชัด ๆ ตรงกลางภาพ และมีแสงสว่างเพียงพอเพื่อผลลัพธ์ที่ดีที่สุด
"""

# จัดรูปเล่มหน้าเว็บให้บังคับรีเฟรชค่ากล่อง Markdown เสมอ
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Mango Leaf Image"),
    outputs=[
        gr.Label(num_top_classes=NUM_CLASSES, label="Predictions"),
        gr.Markdown(value=" รออัปโหลดรูปภาพ...", label="Notice System") # 🌟 ระบุค่า value เริ่มต้นกันมันจำแคชเอ๋อ
    ],
    title="Mango Leaf Cultivar Classifier",
    description=APP_DESCRIPTION,
)

if __name__ == "__main__":
    iface.launch()
