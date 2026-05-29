import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 1. Cấu hình trang web
st.set_page_config(page_title="Nhận Diện Tiền Việt Nam", page_icon="💵")
st.title("💵 Ứng dụng Nhận Diện Tiền Giấy Việt Nam")
st.write("Vui lòng tải lên một bức ảnh tờ tiền để AI dự đoán mệnh giá!")

# 2. Tải mô hình (chỉ tải 1 lần để web không bị chậm)
@st.cache_resource
def load_ai_model():
    # Đảm bảo tên file này khớp với file h5 bạn để trong thư mục
    return load_model("banknote_model_V2.h5")

model = load_ai_model()

# 3. Tạo từ điển dịch kết quả (LƯU Ý QUAN TRỌNG)
# Bạn phải sửa danh sách này cho khớp chính xác với Tên Thư Mục trên Drive của bạn
# sắp xếp theo thứ tự bảng chữ cái alphabet (A-Z, 0-9)
class_names = ['100k', '10k', '20k', '500k', '50k'] 

# 4. Tạo nút tải ảnh lên
uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh người dùng vừa tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh bạn vừa tải lên', use_column_width=True)
    st.write("Đang phân tích...")

    # Tiền xử lý ảnh (giống hệt Bước 5 trên Colab)
    # Chuyển ảnh về RGB để tránh lỗi nếu ảnh là PNG có nền trong suốt
    image = image.convert('RGB')
    image = image.resize((128, 128)) 
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_money = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100

    # In kết quả ra màn hình thật đẹp
    st.success(f"🎉 Dự đoán: Đây là tờ **{predicted_money}**")
    st.info(f"Độ tự tin của AI: {confidence:.2f}%")