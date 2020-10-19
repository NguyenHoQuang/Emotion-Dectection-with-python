from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os


num_classes = 5 # 5 kinds of emotions: happy, neutral, sad, angry, surprise

img_rows,img_cols = 48,48   #reural network: image : 48x48
batch_size = 32 #Do data rất lớn nên ta dùng 32 ảnh để huấn luyện, ta chia thành những batch nhỏ, mỗi patch gồm 32 ảnh

train_data_dir = '/D:/Data/UIT_Project/Emotion Detect with Python/train'
validation_data_dir = '/D:/Data/UIT_Project/Emotion Detect with Python/validation'

#Tiền xử lý ảnh
#Tạo thêm dữ liệu để train
#Generate batches of tensor image data with real-time data augmentation.
#The data will be looped over (in batches).

train_datagen = ImageDataGenerator(
					rescale=1./255,     #Thay đôi kích thước
					rotation_range=30,  # xoay ảnh 30 độ để tạo ra nhiều ảnh có chiều khác nhau được tạo ra từ ảnh gốc
					shear_range=0.3,    # Cắt 30%
					zoom_range=0.3, #zoom 30%
					width_shift_range=0.4,  #Các giá trị theo chiều rộng của float trong khoảng [-0.4;0.4]
					height_shift_range=0.4, #Các giá trị theo chiều cao của float trong khoảng [-0.4;0.4]
					horizontal_flip=True, #Ngẫu nhiên lật đầu vào theo chiều ngang
					fill_mode='nearest') #Các điểm nằm ngoài ranh giới của đầu vào được điền theo chế độ đã cho:aaaaaaaa | abcd | dddddddd

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir, #Data input
					color_mode='grayscale',     #Bởi vì kho dữ liệu train của chúng ta là ảnh xám nên ta chỉnh chế đố grayscale
#Do chúng ta cần cấu trúc của ảnh và các đặt điểm trên khuôn mặt, nên chúng ta ko cần màu, màu xám phù hợp cho việc nhận dạng ảnh

					target_size=(img_rows,img_cols), #Kích thước mà tất cả các hình ảnh tìm thấy sẽ được thay đổi kích thước.
					batch_size=batch_size, #Kich thước của lô dữ liệu dùng để train
					class_mode='categorical', #Chúng ta có nhiều lớp,  5 cảm xúc là 5 lớp
#Mặc định: "categorical". Xác định loại mảng nhãn được trả về: - "categorical" sẽ là nhãn được mã hóa một chiều 2D
					shuffle=True) #Chúng ta cần XÁO TRỘN dữ liệu, để đảm bảo rằng kết quả được chính xác nhất, không có sự sắp đặt từ trước



validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


model = Sequential()
#A Sequential model là mô hình phù hợp cho plain stack của layers, nơi mà mỗi layers có chính xác 1 cái input tensor và 1 cái output tensor

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
#32: chiều của không gian đầu ra (tức là số lượng bộ lọc đầu ra trong tích chập).
#padding "same"dẫn đến việc đệm đều ở bên trái / phải hoặc lên / xuống của đầu vào sao cho đầu ra có cùng chiều cao / chiều rộng với đầu vào.
model.add(Activation('elu'))
#ELU: Các đơn vị tuyến tính theo cấp số nhân cố gắng làm cho các kích hoạt trung bình gần bằng 0, giúp tăng tốc độ học tập.
# Nó đã được chứng minh rằng ELU có thể có được độ chính xác phân loại cao hơn ReLU
model.add(BatchNormalization())     #Chuẩn hóa hàng loạt
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Lớp Dropout ngẫu nhiên đặt các đơn vị đầu vào thành 0 với tần suất rate ở mỗi bước trong thời gian đào tạo, giúp ngăn ngừa quá mức.
#   Các đầu vào không được đặt thành 0 được tăng lên 1 / (1 - tỷ lệ) sao cho tổng trên tất cả các đầu vào không thay đổi.

# Block-2

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5
#Fully Connected Layer
model.add(Flatten())
#Làm phẳng đầu vào. Không ảnh hưởng đến kích thước lô.
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6
#Fully Connected Layer
model.add(Dense(64,kernel_initializer='he_normal'))
#Cho phép kết nối dày đặt tại lớp Neural Network
#64: chiều của không gian đầu ra.
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))
#Softmax chuyển đổi một vectơ thực thành một vectơ xác suất phân phân phối.
#Softmax thường được sử dụng làm kích hoạt cho lớp cuối cùng của mạng cnn,  vì kết quả đc hiển thị là phân phối xác suất.

print(model.summary())


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#Callback to save the Keras model or model weights at some frequency
checkpoint = ModelCheckpoint('/D:/Data/UIT_Project/Emotion Detect with Python/Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

#Dừng đào tạo khi một số liệu được theo dõi đã ngừng cải thiện.
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

#Reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2, #yếu tố mà tỷ lệ học tập sẽ giảm
                              patience=3, #số lượng epochs không có cải thiện sau đó tốc độ học tập sẽ giảm.
                              verbose=1, #0: yên lặng, 1: cập nhật tin nhắn.
                              min_delta=0.0001) #ngưỡng để đo tối ưu mới, chỉ tập trung vào những thay đổi quan trọng.

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)























































