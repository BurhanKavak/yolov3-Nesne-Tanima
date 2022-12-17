# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:04:36 2022

@author: burha
"""
#%% 1. Bölüm

import cv2
import numpy as np

# Yolonun sitesinden hazır eğitilmiş model kullanıyoruz YOLOv3-416 CFG ve WEIGHTS dosyalarını indiriyoruz.


# resmi okumamız için kullanıyoruz.
img = cv2.imread("images/people.jpg")


# image (748,1000,3) boyutunda olduğundan genişlik yükseklik bilgilerini bu şekilde alıyoruz
img_width = img.shape[1]
img_height = img.shape[0]


#%% 2.Bölüm

# resmi 4 boyutlu tensörlere çeviriyoruz (blob format)
img_blob = cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB = True,crop = False)


# modelimizin tanıdığı nesnelere göre etiketleme yapıyoruz (toplam 80 adet nesneyi tanımaktadır)
labels = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]


# her bir label için farklı renk oluşturup bounding box'ları ayrıştırmaya yarar
colors = ["255,0,0","0,255,255","150,0,240","255,255,0","0,255,0"] # girdiğim renklere göre ayıracak bounding boxları
colors = [np.array(color.split(",")).astype("int") for color in colors] # colors arrayini tek tek dolaşıp virgüle göre ayırıyoruz ve tipini int yapıyoruz.
colors = np.array(colors) # array formatına çeviriyorum ki daha düzenli gözüksün.
colors = np.tile(colors,(18,1)) # ilk değer alt alta kaç tane eklemek istediğimiz ikinci değer ise yan yana kaç tane eklemek istediğimiz.


#%% 3.Bölüm

# cfg ve weight dosyalarımı model içinde tutuyorum
model = cv2.dnn.readNetFromDarknet("C:/Users/burha/YoloKursu/pretrained_model/yolov3.cfg","C:/Users/burha/YoloKursu/pretrained_model/yolov3.weights")

layers = model.getLayerNames() # Modelin içindeki katmanlara ulaşabilmek için kullanıyoruz. Burada bazı katmanlar bizim output değerimiz olacak.
# model.getUnconnectedOutLayers() metodunu kullandığımızda bize hangi indekslerde output katmanı olduğunu veriyor ama indeksler 0 dan başladığı için biz bunu 1 eksiği şeklinde arayacaz.
# örnek verecek olursak 200. indeksde bir yolo output var ise bunun değeri 199'da gözükür.
output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob) # model bizden 4 boyutlu bir tensör istiyordu onu oluşturmuştuk ve içine attık
detection_layers = model.forward(output_layer)

#%% 4.Bölüm

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        #skor için kullanılıyor ilk 5 değer bounding box'a ayrılmaktadır.
        scores = object_detection[5:]
        predicted_id = np.argmax(scores) # Maksimum değerinin bulunduğu indeksi getiriyor
        confidence = scores[predicted_id] # güven skorunun tutulduğu yer
        
        if confidence > 0.90: # güven skorum %90 dan büyükse bounding box'ı oluştursun
            
            label = labels[predicted_id] 
            bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            (box_center_x,box_center_y,box_width,box_height) = bounding_box.astype("int")
            
            star_x = int (box_center_x - (box_width / 2))
            star_y = int (box_center_y - (box_height / 2))
            
            end_x = star_x + box_width
            end_y = star_y + box_height
            
            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]
            
            
            
            cv2.rectangle(img,(star_x,star_y),(end_x,end_y),box_color,1)
            cv2.putText(img,label,(star_x,star_y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5, box_color, 1)


while True:
    
    cv2.imshow("Detection Window",img)
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()




