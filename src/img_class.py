import os
import shutil
from scipy.io import loadmat

# Đường dẫn đến các thư mục của bạn
data_dir = 'data' # Thay đổi đường dẫn này
train_dir = os.path.join(data_dir, 'cars_train')
devkit_dir = os.path.join(data_dir, 'car_devkit')
output_train_dir = os.path.join(data_dir, 'train_organized')

# 1. Đọc file metadata để lấy tên xe
meta = loadmat(os.path.join(devkit_dir, 'cars_meta.mat'))
class_names = [name[0] for name in meta['class_names'][0]]

# 2. Đọc file annotation của tập train
train_annos = loadmat(os.path.join(devkit_dir, 'cars_train_annos.mat'))['annotations'][0]

# 3. Tạo thư mục và copy ảnh vào đúng loại
for anno in train_annos:
    file_name = anno[-1][0]
    label_id = anno[-2][0][0]
    class_name = class_names[label_id - 1].replace(' ', '_').replace('/', '-')
    
    # Tạo thư mục cho từng loại xe nếu chưa có
    target_folder = os.path.join(output_train_dir, class_name)
    os.makedirs(target_folder, exist_ok=True)
    
    # Copy ảnh từ folder gốc vào folder tên xe
    src_path = os.path.join(train_dir, file_name)
    dst_path = os.path.join(target_folder, file_name)
    shutil.copy(src_path, dst_path)

print("Đã tổ chức xong dữ liệu!")