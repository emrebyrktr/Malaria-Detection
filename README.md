Malaria Detection with DL


# Malaria Detection with Deep Learning Models

Bu proje, üç farklı hücresel görüntü veri seti kullanılarak sıtma enfeksiyonunun otomatik olarak tespit edilmesini amaçlayan bir derin öğrenme çalışmasını içerir. Her veri seti için ayrı bir model eğitilmiş ve sonuçlar doğruluk oranları üzerinden değerlendirilmiştir.


---

## 📁 Kullanılan Veri Setleri

Bu projede üç farklı veri seti kullanılmıştır. Her veri setine ait örnek görüntüler aşağıda verilmiştir:

1. Veri Seti – Chittagong Medical College Hospital (Bangladeş)
<img width="482" height="695" alt="image" src="https://github.com/user-attachments/assets/7a030e5b-1aa0-44dc-9ec5-939542304b89" />

Bu veri seti, Bangladeş Chittagong Tıp Fakültesi Hastanesi’nde takip edilen 150 doğrulanmış sıtma vakasına ait mikroskop görüntülerinden oluşmaktadır. Her görüntü, uzmanlar tarafından etiketlenmiş olup enfekte ve enfekte olmayan hücre yapılarını yüksek çözünürlükte içermektedir. Klinik ortamda elde edildiği için hücre morfolojileri gerçek hasta verilerini yansıtmaktadır.

2. Veri Seti – Trakya Üniversitesi Mikrobiyoloji Laboratuvarı
<img width="449" height="345" alt="image" src="https://github.com/user-attachments/assets/a80a138e-94fa-489f-908b-4107917d0320" />

Bu veri seti, Trakya Üniversitesi Mikrobiyoloji Bölümü tarafından çeşitli klinik kan örneklerinden hazırlanmış mikroskop görüntülerini içermektedir. Farklı laboratuvar koşullarında boyanmış ve hazırlanmış hücre örnekleri sayesinde görüntüler, renk yoğunluğu ve hücresel yapı açısından çeşitlilik göstermektedir. Bu, modelin gerçek dünya senaryolarında genelleme kabiliyetini artırmaktadır.

3. Hibrit Veri Seti – Birleştirilmiş Multikaynak Görüntü Havuzu

Üçüncü veri seti, birinci ve ikinci veri setlerinin birleştirilmesiyle oluşturulmuş hibrit bir görüntü havuzudur. Bu birleşik veri seti, iki farklı coğrafi bölge ve iki farklı laboratuvar ortamından elde edilen mikroskop görüntülerini içerdiği için daha geniş varyasyon, daha fazla sınıf dengesi ve modelin daha sağlam genelleme performansı için ideal bir yapı sunar.


Bu çalışma sıtmanın bir türü olan Plasmodium Falciparum türüne aittir.



## 🧠 Kullanılan Modeller

Her veri seti için 3 ayrı model eğitildi. Performans sonuçları karşılaştırıldı.

1. CNN (Convolutional Neural Network)

Klasik evrişimsel sinir ağı mimarisi. Hafif, hızlı ve düşük parametreli bir yapı sunar.

Eğitildiği veri setleri:

Veri Seti 1: Chittagong Medical College Hospital

Veri Seti 2: Trakya Üniversitesi Mikrobiyoloji Laboratuvarı

Veri Seti 3: Hibrit veri seti

2. VGG-16 Wft

VGG-16 modeli denenmiştir. VGG-16 modeli eğitilirken fine tuning işlemi uygulanmıştır.

Eğitildiği veri setleri:

Veri Seti 1: Chittagong Medical College Hospital

Veri Seti 2: Trakya Üniversitesi Mikrobiyoloji Laboratuvarı

Veri Seti 3: Hibrit veri seti

3. CNN-ViT (Hybrid Model)

CNN ve Vision Transformer mimarilerinin güçlü yönlerini bir araya getiren bir hibrit modeldir.

Eğitildiği veri setleri:

Veri Seti 1: Chittagong Medical College Hospital

Veri Seti 2: Trakya Üniversitesi Mikrobiyoloji Laboratuvarı

Veri Seti 3: Hibrit veri seti

## 📊 Sonuçlar (Sadece Accuracy - Makale nedeniyle sınırlı paylaşım)
Her veri setine ait en iyi doğruluk sonucu paylaşılmıştır.

Veri Seti 1'e ait en yüksek doğruluk oranı VGG-16 Wft modeline aittir. %97,06' lık doğruluk oranına ulaşılmıştır.

Veri Seti 2'e ait en yüksek doğruluk oranları her modelde aynı gelmiştir. %98,18 'lik doğruluk oranına ulaşılmıştır.

Veri Seti 3'e ait en yüksek doğruluk oranı VGG-16 Wft modeline aittir. %96,85 'lik doğruluk oranına ulaşılmıştır.

> Not: Diğer metrikler (loss, precision, recall, F1-score, MCC vs.) makale yayını nedeniyle paylaşılmamıştır.

---
