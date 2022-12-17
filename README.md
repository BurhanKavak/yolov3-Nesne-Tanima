# yolov3-Nesne-Tanıma
### Yüklediğimiz fotoğraftaki objeleri tanımaya çalışan basit bir nesne tespiti algoritması kullanıyoruz.
#### Kullanacağımız COCO Dataset 416'dır buradaki CFG ve WEIGHTS dosyalarını indirip bunun yolunu belirtmeniz gerekmektedir.
#### Yazılan kodlar 416 için yazılmıştır kendinize göre bireyselleştirebilirsiniz.
#### https://pjreddie.com/darknet/yolo/ sitesinden ulaşabilirsiniz.
![1](https://user-images.githubusercontent.com/79043326/208213674-fcea2c55-6614-4c7e-82c8-03fed89564ed.png)
![2](https://user-images.githubusercontent.com/79043326/208213733-ed99479f-85a2-4d04-b8a3-cf640539db66.png)


## Nesneleri tespit etmede sıkıntı yaşamıyor ama güven skorunu %90 üzerinde almış olsak bile tek bir nesne için birden fazla tanımaya çalışıyor ve fazladan bounding box yapıyor bu istediğimiz bir durum değil bunun için Non-Maximum Suppression kullanılmaktadır. Kod için bu optimizasyon işlemi yapılmamıştır.



![4](https://user-images.githubusercontent.com/79043326/208214298-18bb2d89-c56b-4a64-9e43-1c5e428d8f94.png)
