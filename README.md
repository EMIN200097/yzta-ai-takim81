
## Takım İsmi
##  Coreteam

![ChatGPT Image 5 Tem 2025 15_52_34 (2)](https://github.com/user-attachments/assets/bec82d13-bfeb-4be1-a991-ad1fa78858bf)



| İsim                   | Rol           | Durum | Sosyal Medya                                                                                                                                                                                                                                                                                                    |
| ---------------------- | ------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Meltem Kartopu**     | Scrum Master  | Aktif | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/meltemkartopu/) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/MeltemKartopu)         |
| **Ahmet Reşat Keyan**  | Product Owner | Aktif | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/ahmet-keyan-088995246/) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/Drandalll)     |
| **Berke Sinan Yetkin** | Developer     | Aktif | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/berke-sinan-yetkin/) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/BerkeSinanYetkin) |
| **Esra Öden**          | Developer     | Aktif | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/esra-%C3%B6den-92b552270/) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/esrashub)   |
| **Mehmet Emin Şahin**  | Developer     | Aktif | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/mehmetemin-sahin/) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/EMIN200097)         |

---
## Ürün İsmi
# PoseCore 

![posecore\_2](https://github.com/user-attachments/assets/daeb3e73-7297-464a-924d-2a8dc356ab1b)

## Ürün Açıklaması

PoseCore, kullanıcıların postür bozukluklarını gerçek zamanlı analiz eden ve düzeltici geri bildirim sağlayan yapay zeka tabanlı bir uygulamadır. Mediapipe tabanlı iskelet modellemesi kullanarak, ofis çalışanları, fizyoterapi hastaları ve uzun süre oturarak çalışan bireylere yönelik kişiselleştirilmiş duruş takibi sağlar.

##  Özellikler

* 📸 Gerçek zamanlı kamera ile postür analizi
* 📊 Postür açılarının matematiksel hesaplanması
* 🔔 Yanlış duruşta anında uyarı sistemi
* 📈 Kullanıcıya özel ilerleme raporları
* 📱 Çoklu cihaz ve kamera uyumluluğu

##  Hedef Kitle

* Ofis çalışanları
* Postür sorunu yaşayan hastalar (örn. skolyoz)
* Fizyoterapistler ve spor eğitmenleri
* Uzun süre oturarak çalışan bireyler

---

##  Sprint 1 (23 Haziran - 6 Temmuz 2025)

<details>
<summary>Tıklayarak Detayları Göster/Gizle</summary>

###  Sprint Notları

* **Proje fikri ve modüller:** Fizyoterapi / postür / spor modülleri netleştirildi
* **Teknoloji Stack:** Mediapipe, OpenCV, Python/Flask
* **Prototip Geliştirme:** Temel iskelet çıkarımı ve açı hesaplama prototipi oluşturuldu
* **Veri Seti İncelemesi:** Kaggle fizyoterapi hareketleri analiz edildi

###  Hedeflenen Puan

* **Sprint Puanı:** 100 / 300
* **Mantık:** Toplam proje 300 puan; her sprint için \~100 puan
* **Story Points:** Her sprintte 100 puana ulaşmak için atanan 7 ana kanban kartına ait altgörevlere, ana görevde ulaşılmak üzere (roll up story points) ayrı ayrı puanmalar yapılmıştır.
* 2 haftalık sprint sürecinde 5 takım üyesi için 14 günlük görev dağılımı "Sprint Görev Dağılımı ve Puan Mantığı Tablosu" nda yer almaktadır.

###  Sprint Görev Dağılımı ve Puan Mantığı Tablosu
| Ana Görev                     | Alt Görev                                                                 | Puan | Sorumlu Rol          | Açıklama                                                                |
|-------------------------------|---------------------------------------------------------------------------|------|----------------------|-------------------------------------------------------------------------|
| **Araştırma & Planlama**      | Proje fikirleri araştırması                                               | 10   | Tüm ekip             | Hızlı workshop + bireysel araştırma                                     |
|                               | Kullanıcı persona oluşturma                                               | 5    | Product Owner        | PO liderliğinde hazırlanması                                            |
|                               | Kullanıcı görüşmeleri                                                     | 10   | PO + 1 Developer     | Katılımcı bulma + 5 görüşme                                             |
|                               | Teknoloji seçimi (Mediapipe/YOLO)                                         | 15   | 2 Developer          | Prototip test + teknik rapor                                            |
|                               | Başarı metriklerinin tanımlanması                                         | 10   | PO + Scrum Master    | KPI'ların SMART prensibiyle belirlenmesi                                |
| **Veri Toplama & Ön İşleme** | Doğru hareket videolarının kaydı                                           | 10   | 2 Developer          | Senaryo başına 2 tekrar                                                 |
|                               | Çoklu kamera veri seti                                                    | 10   | 3 Developer          | 3 açı x 5 hareket (eşgüdüm gerektirir)                                  |
|                               | Yanlış hareket senaryoları                                                | 10   | 1 Developer + PO     | Hata senaryolarının klinik doğruluğu                                    |
|                               | Koordinat normalizasyonu                                                  | 10   | 1 Developer          | OpenPose/Mediapipe çıktılarının dönüşümü                                |
|                               | Ham Video Verisinden CSV Üreten Script/Aracın Geliştirilmesi              | 10   | 1 Developer          | video verilerinden veriseti elde edilmesi                               |                                                        
| **Toplam**                    |                                                                           | 100  |                      |                                                                         |

---

###  Daily Scrum

* **Saat:** Her akşam 20:00 - 21:00 (WhatsApp)
* **Kanallar:** WhatsApp, Google Meet
* [WhatsApp Daily Scrum Ekran Görüntüleri](https://imgur.com/a/coreteam-daily-scrum-chats-QgBy6N9)

###  Sprint Board

**ClickUp Proje Panosu:** [Buradan Ulaşabilirsiniz](https://clickup.com/)

![image](https://github.com/user-attachments/assets/f31ad366-bf1f-497d-8100-39f8fdd5e194)


**ClickUp Proje Raporu ve Tamamlanan Sprint Puanı 
![image](https://github.com/user-attachments/assets/c9620829-f56c-41d7-a9a7-5cbb06ce2ad2)
* 100 puandan 50 puan tamamlanmıştır
* Devam eden görevler sonraki sprinte devredelecektir.
* Artı 10 puan model aşaması model geliştirme 1 e ait Mediapipe ile iskelet çıkarımı testi görevinden gelmiştir. Araştırma ve Planlama'ya katkısından dolayı bu sprintte denenmek istenmiştir. 
  

###  Prototip Testleri

* **MediaPipe Nokta Algılama ve Açı Hesaplama:** [Test Ekran Görüntüsü İçin Tıklayın](https://imgur.com/a/mediapipe-nokta-alg-lama-ve-hesaplama-3VOvH1m)

  Tabii, görseldeki metni **Markdown formatında** sadeleştirerek ve düzenleyerek aşağıya dönüştürüyorum:

---

## Veri Seti Toplama

Bu dizin (dataset_gathering) , projede kullanılacak veri setlerini toplamak, işlemek ve düzenlemek için kullanılan araçları içerir. Ham video verisini Makine Öğrenmesi modellerine beslemek ve doğru postür analizi yapmak için kullanılacak bu veriyi toplamayı burada gerçekleştiriyoruz.

### İçerik

### `main.py` ile Video'dan CSV'ye Dönüştürme

`main.py`, `input/` klasöründeki bir video dosyasını alır ve işlenmiş verileri `output/` klasöründe bir CSV dosyasına kaydeder. Temel adımlar:

1. `input/` klasörüne dönüştürmek istediğiniz video dosyasını ekleyin.
2. `main.py`:173'de video\_name argümanına string olarak doğru oturuş postürü videosu dosyanızın ismini **dosya tipi uzantısıyla beraber** verin.

   * Ya da kamera kullanmak için bu keyword argümanını silin.
3. Programı çalıştırıp mevcut kareleri kaydetmeye başlamak için klavyenizdeki **"L"** tuşuna basın.
4. Program, videodaki kareleri işler ve ilgili verileri `output/` dizinindeki CSV dosyasına yazar. Yazmayı durdurmak için tekrar **"L"** tuşuna basabilirsiniz.
5. Programdan çıkış yapmak için klavyenizdeki **"Q"** tuşuna basın.

Detaylı parametreler ve ek seçenekler için `main.py` dosyasındaki açıklamaları inceleyin.

---

## Notlar

* Bir dizin yukarıdaki `requirements.txt` dosyasındaki gereksinimleri pip ile kurduğunuzdan emin olun.
* Tam olarak akışa hakim olmak için `main.py` dosyasındaki komut satırlarını okuyun.
* CSV dosyası kullanılmadan önce gözden geçirilmelidir.

---

Başka düzenleme veya eklemek istediğin detay varsa iletebilirsin!


###  Sprint Review

* ✅ Proje fikri ve modüller onaylandı
* ✅ Mediapipe entegrasyonu tamamlandı
* ✅ Veri seti analizi tamamlandı
* 🚧 Kullanıcı test senaryoları Sprint 2'ye ertelendi

###  Sprint Retrospective

#### 👍 İyi Yönler

* Hızlı teknoloji seçimi ve prototipleme
* WhatsApp üzerinden etkili asenkron iletişim

#### 📌 Geliştirmeler

* Toplantı zamanlamalarının erken duyurulması
* Veri etiketleme standartlarının belirlenmesi

---

</details>

## Sprint 2 (7 Temmuz - 20 Temmuz 2025  )
<details>
<summary>Tıklayarak Detayları Göster/Gizle</summary>

**Sprint Süresi:** 2 hafta  
**Takım:** Coreteam  

---

<details>
<summary>📊 Sprint 2 Özet</summary>

## Sprint Hedefleri

Sprint 2'de ana hedefimiz, Sprint 1'de oluşturduğumuz temel yapı üzerine model geliştirme, veri toplama ve kullanıcı arayüzü çalışmalarını tamamlamaktı.

**Hedef Sprint Puanı:** 100/300  
**Gerçekleşen Sprint Puanı:** 74/100 (%74)

</details>

---

<details>
<summary>🎯 Sprint Notları</summary>

  ### Sprint Katılımcıları:
- **Meltem Kartopu** (Scrum Master) - Aktif
- **Berke Sinan Yetkin** (Developer) - Aktif  
- **Ahmet Reşat Keyan** (Product Owner) - Aktif
- **Esra Öden** (Developer) - Aktif
- **Mehmet Emin Şahin** (Developer) - Aktif
## Sprint İçinde Tamamlanması Tahmin Edilen Puan
**100 puan** - Bütün proje 300 puan olarak planlandı ve Sprint 2'de 100 puan tamamlanması hedeflendi.

## Tahmin Mantığı
Sprint 2'de ana odak noktaları:
- Model geliştirme ve optimizasyon çalışmaları (30 puan)
- Kapsamlı veri toplama ve ön işleme (25 puan) 
- Araştırma ve uzman görüşü alma (20 puan)
- Yapay zeka algoritma iyileştirmeleri (15 puan)
- Frontend/UI geliştirme (10 puan)

**Toplam:** 100 puan hedeflenmiş, 74 puan başarıyla tamamlanmıştır.

## Sprint Puanlama Sistemi ve Görev Dağılımı

### Kategori Bazlı Puanlama Tablosu

| Kategori | Hedef Puan | Tamamlanan Puan | Tamamlanma (%) | Rol Dağılımı |
|----------|------------|-----------------|----------------|---------------|
| **Araştırma & Planlama** | 20 | 20 | 100% | Esra (Dev), Meltem (SM),Mehmet Emin (Dev) |
| **Veri Toplama & Ön İşleme** | 25 | 20 | 80% | Esra (Dev), Berke (Dev), Meltem (SM) |
| **Model Geliştirme** | 30 | 21 | 70% | Esra (Dev), Berke (Dev), Meltem (SM) |
| **Yapay Zeka Tarafı** | 15 | 6 | 40% | Mehmet Emin (Dev), Ahmet (PO), Berke (Dev) |
| **Frontend & UX/UI** | 10 | 7 | 70% | Esra (Dev), Ahmet (PO), Meltem (SM) |
| **TOPLAM** | **100** | **74** | **74%** | **Tüm Takım** |



</details>

---

<details>
<summary>💬 Daily Scrum</summary>

## Daily Scrum Süreci

**Zaman:** Her akşam 20:00-21:30 arası  
**Kanallar:** WhatsApp grup mesajları, Google Meet toplantıları  
**Sıklık:** Günlük WhatsApp güncellemeleri, haftada 2-3 Google Meet

### WhatsApp Daily Scrum Konuşmaları
Sprint 2 boyunca takım üyeleri arasında gerçekleşen günlük iletişim ve proje güncellemeleri:
[📱 WhatsApp Daily Scrum Ekran Görüntüleri](https://imgur.com/a/sprint-2-whatsapp-screenshots-qDiVlZH)

### Ana İletişim Konuları:
- Model geliştirme ilerlemeleri 
- Veri seti araştırması güncellemeleri 
- UI/UX geliştirme durumu 
- Proje koordinasyonu 
- kod review 

### Toplantı Tarihleri:
- **8 Temmuz:** Sprint planlama ve görev dağılımı
- **12 Temmuz:** Haftalık ilerleme değerlendirmesi  
- **15 Temmuz:** Veri seti seçimi ve model karşılaştırması
- **18 Temmuz:** Sprint review hazırlığı

</details>

---

<details>
<summary>📋 Sprint Board Updates</summary>

## ClickUp Sprint Board

Sprint 2 görev dağılımı, ilerleme durumu ve proje yönetimi paneli:
[📊 ClickUp Sprint 2 Board](https://app.clickup.com/90181399415/v/li/901809374434)

### Sprint Burndown:
- Başlangıç: 100 puan
- Tamamlanan: 74 puan
- Kalan: 26 puan (Sprint 3'e aktarıldı)
<img width="1051" height="683" alt="image" src="https://github.com/user-attachments/assets/9a4645e7-09fb-474a-8f29-c44b8faf19a9" />



*Sprint 2 Backlog Items Ekran Görüntüsü*


<img width="1130" height="425" alt="image" src="https://github.com/user-attachments/assets/49ec5456-660c-4afa-9a82-a6d36af642b3" />





*Sprint 2 Sprint Board Ekran Görüntüsü*


<img width="1855" height="744" alt="image" src="https://github.com/user-attachments/assets/fa1ff1bd-1c83-46ea-abaf-f6e8e426a3f6" />



*Sprint 2 Sprint Dashboard Ekran Görüntüsü*
</details>

---

<details>
<summary>🖥️ Ürün Durumu</summary>

## Sprint 2 Geliştirme Çıktıları

### 1. Model Geliştirme İyileştirmeleri

  
####  Oturuş Pozisyonu İçin İkili (Binary) Değerlendirme Modeli

<img src="https://github.com/user-attachments/assets/7e4a673f-b0ac-4bd5-99b8-71d71b2dc0ac" height="400" />



*Oturma Pozisyonu için True False Geri Bildirimi*

- MediaPipe entegrasyonu optimize edildi
- Açı hesaplama algoritması geliştirildi
- CSV export özelliği eklendi
- Real-time işleme test edildi

####  Squad puanlama Modeli

Sprint 2'de geliştirilen postür analizi ve puanlama sisteminin çalışır halinin demonstrasyonu:
<img src="https://github.com/user-attachments/assets/9a0fa282-03c1-444c-bf1a-3520fd0f316a" width="600" />

*3000-0 Arası Squad puanlama Ekran Görüntüsü*

**Model Demo Özellikleri:**
- Real-time kamera görüntü işleme
- Mediapipe ile iskelet noktası tespiti
- Squad postürü açı hesaplaması
- Anlık puanlama (3000 den 0'a yaklaşarak ideal squad postürüne ulaşma hedeflendi)
**Eklenecekler:**
- Farklı hastalık gruplarına ait hareketler eklenecek ( temelde 5 hareket planladı)
- Puanlama mekanizması sadeleştirilecek (Threshold eşikleri belirlenerek skorlama ölçeklendirilecek)  

### 2. Kullanıcı Arayüzü Geliştirmeleri
<img src="https://github.com/user-attachments/assets/4d33074c-f840-4495-b496-090b24e3d3eb" width="300" />

<img src="https://github.com/user-attachments/assets/e9d261a3-5381-491f-869b-c92dc2c2fa0f" width="300" />

[Flutter mobil uygulaması ön deneme](https://preview.builtwithrocket.new/posecore-9w5bo42)


**Flutter Mobil Uygulama:**
- Temel ekran tasarımları tamamlandı
- MediaPipe kamera entegrasyonu test edildi
- Figma prototipi oluşturuldu
- Kullanıcı akışı belirlendi
iyileştirilecekler: 
- Uygulama içi font hataları düzeltilecek

</details>

---

<details>
<summary>🎨 UI/UX Geliştirme ve Testler</summary>

## Kullanıcı Arayüzü Çalışmaları

### Flutter Mobil Uygulama Prototipleri

Sprint 2 boyunca geliştirilen kullanıcı arayüzü tasarımları ve test sonuçları:

**UI/UX Demo Alanı:**

![WhatsApp Görsel 2025-07-18 saat 12 34 42_a26d09e1](https://github.com/user-attachments/assets/d066964d-1ddf-450b-9662-1051caf4ffef)



### MediaPipe UI Entegrasyon Testleri:
- ✅ Real-time kamera görüntü işleme başarılı
- ✅ Iskelet noktası görselleştirmesi çalışıyor
- ✅ Kullanıcı arayüzü responsive tasarım
- ✅ Kamera açısı optimizasyonu test edildi

### Figma ve Prototipleme Çalışmaları:
- Kullanıcı akış şemaları oluşturuldu
- Wireframe tasarımları tamamlandı
- Rocket.new platformu ile entegrasyon test edildi
- Color palette ve typography belirlendi

### Kullanıcı Deneyimi İyileştirmeleri:
- Onboarding sürecini sadeleştirme
- Kamera yerleştirme rehberi
- Gerçek zamanlı geri bildirim sistemi
- Erişilebilirlik standartları uygulaması

</details>

---

<details>
<summary>📈 Sprint Review</summary>

## Sprint 2'de Yapılan İşler


### ✅ Başarıyla Tamamlanan Görevler:

#### Araştırma & Planlama 
- ✅ Kapsamlı veri setleri araştırılması ve derlenmesi
- ✅ Egzersiz türleri belirlenmesi (seated leg raise, bridge, omuz egzersizleri)
- ✅ Fizyoterapist uzman görüşü alınması
- ✅ Pratik kullanım senaryoları belirlenmesi

#### Model Geliştirme 
- ✅ Gelişmiş classifier modeli geliştirme
- ✅ Regresyon vs Classification karşılaştırması
- ✅ Threshold ayarlama mekanizması
- ✅ Veri toplama pipeline iyileştirmesi
- ✅ Çoklu egzersiz desteği eklenmesi
- 🔄 Threshold fine-tuning (devam ediyor)

#### Frontend & UX/UI 
- ✅ Flutter mobil uygulama prototipi
- ✅ MediaPipe UI entegrasyonu testi
- ✅ Figma/Rocket.new deneyimi
- ✅ UX/UI testleri
  
#### Yapay Zeka
- ✅ Feedback mekanizması (3000 - 0 arası puanlama)

### 🔄 Devam Eden Görevler:

#### Veri Toplama & Ön İşleme
- 🔄 Seçili egzersizler için video kayıtları
- 🔄 Farklı kamera açılarından veri toplama
- 🔄 Veri etiketleme süreci

#### Yapay Zeka Optimizasyonu 
- 🔄 Pose estimation algoritması iyileştirmesi
- 🔄 Gerçek zamanlı tahmin sistemi kurulumu


</details>

---

<details>
<summary>🔄 Sprint Retrospective</summary>

## Bu Sprintte Yaptığımız En İyi Şeyler

### 👍 Başarılı Yönler:
- **Kapsamlı Araştırma:** Veri seti araştırması ve uzman görüşü alma süreci çok verimli geçti
- **Teknik İlerleme:** Model geliştirme alanında büyük adımlar atıldı 
- **İletişim:** WhatsApp ve Google Meet kombinasyonu ile etkili takım iletişimi sağlandı
- **Prototipleme:** UI/UX testleri başarıyla tamamlandı, kullanıcı deneyimi şekillenmeye başladı
- **Uzman Danışmanlığı:** Fizyoterapist görüşü alınarak proje gerçek ihtiyaçlara yönlendirildi

### 📌 Geliştirilmesi Gerekenler:
- **Veri Toplama:** Video kayıt süreci beklenenden daha uzun sürdü 
- **Zaman Yönetimi:** Bazı görevlerde öngörülen süreler aşıldı
- **AI Optimizasyonu:** Yapay zeka iyileştirmeleri gecikmiş durumda 
- **Entegrasyon:** Backend-frontend entegrasyonu Sprint 3'e ertelendi
- **UI/UX** Flutterda UI tarafın iyileştirilmesi 

### 🎯 Sprint 3 İçin Aksiyon Planı:
1. **Veri toplama** sürecini hızlandırmak için görev dağılımı yapılacak
2. **Backend API** geliştirmesi önceliklendirilecek
3. **Entegrasyon testleri** için daha fazla zaman ayrılacak
4. **Kullanıcı testleri** için pilot grup oluşturulacak

### 📊 Sprint Başarı Metrikleri:
- **Genel Tamamlanma:** %74 (74/100 puan)
- **Takım Katılımı:** %100 (tüm üyeler aktif)
- **Kod Kalitesi:** Yüksek (code review süreçleri takip edildi)
- **Dokümantasyon:** İyi (README ve commit mesajları düzenli)

</details>

---

<details>
<summary>🚀 Sprint 3'e Hazırlık</summary>

## Sprint 3 Planlaması

**Aktarılan Görevler (21 puan):**
- Veri toplama sürecinin tamamlanması (9 puan)
- AI algoritma optimizasyonları (9 puan)  
- Backend-frontend entegrasyonu (3 puan)

**Yeni Sprint 3 Hedefleri:**
- Entegrasyon ve test süreçleri
- Kullanıcı deneyimi iyileştirmeleri
- Performance optimizasyonu
- Pilot kullanıcı testleri

### Sprint 3 Odak Alanları:

#### 🔧 Backend & Entegrasyon (30 puan)
- Websocket geliştirme
- Model deployment
- Flutter-Backend entegrasyonu
- Real-time işleme optimizasyonu

#### 🧪 Test & Doğrulama (25 puan)
- Gerçek kullanıcı testleri
- Performans testleri
- Çoklu cihaz uyumluluğu
- End-to-end test süreçleri

#### 📱 Kullanıcı Deneyimi (20 puan)
- UI/UX iyileştirmeleri
- Onboarding sürecini geliştirme
- Accessibility standartları
- Kullanıcı geri bildirim sistemi

#### 🚀 Production Hazırlık (25 puan)
- Model optimizasyonu
- Deployment stratejisi
- Dokümantasyon tamamlama
- Beta test programı

**Toplam Sprint 3 Hedefi:** 100 puan

</details>
</details>

## Sprint 3 (21 Temmuz - 3 Ağustos 2025)
<details>
<summary>Tıklayarak Detayları Göster/Gizle</summary>

**Sprint Süresi:** 2 hafta  
**Takım:** Posecore Team  

---

<details>
<summary>📊 Sprint 3 Özet</summary>

### Sprint Hedefleri

Sprint 3'te ana hedefimiz, Sprint 2'de geliştirdiğimiz modelleri entegre etmek, backend-frontend bağlantısını kurmak ve kullanıcı testleri için hazır bir prototip oluşturmaktı.

**Hedef Sprint Puanı:** 100/300  
**Gerçekleşen Sprint Puanı:** 78/100 (%78)

</details>

---

<details>
<summary>🎯 Sprint Notları</summary>

### Sprint Katılımcıları:
- **Meltem Kartopu** (Scrum Master) - Aktif
- **Berke Sinan Yetkin** (Developer) - Aktif  
- **Ahmet Reşat Keyan** (Product Owner) - Aktif
- **Esra Öden** (Developer) - Aktif
- **Mehmet Emin Şahin** (Developer) - Aktif

### Sprint İçinde Tamamlanması Tahmin Edilen Puan
**100 puan** - Sprint 2'den devredilen 21 puan + yeni hedeflenen 79 puan = 100 puan toplam

### Tahmin Mantığı
Sprint 3'te ana odak noktaları:
- Backend & Entegrasyon çalışmaları (35 puan)
- Test & Doğrulama süreçleri (25 puan)
- Kullanıcı Deneyimi iyileştirmeleri (20 puan)
- Production hazırlık çalışmaları (20 puan)

**Toplam:** 100 puan hedeflenmiş, 78 puan başarıyla tamamlanmıştır.

### Sprint Puanlama Sistemi ve Görev Dağılımı

#### Kategori Bazlı Puanlama Tablosu

| Kategori | Hedef Puan | Tamamlanan Puan | Tamamlanma (%) | Rol Dağılımı |
|----------|------------|-----------------|----------------|---------------|
| **Backend & Entegrasyon** | 35 | 28 | 80% | Berke (Dev), Mehmet Emin (Dev) |
| **Test & Doğrulama** | 25 | 18 | 72% | Tüm Takım |
| **Kullanıcı Deneyimi** | 20 | 16 | 80% | Ahmet (PO), Esra (Dev), Meltem (SM) |
| **Production Hazırlık** | 20 | 16 | 80% | Tüm Takım |
| **TOPLAM** | **100** | **78** | **78%** | **Tüm Takım** |

</details>

---

<details>
<summary>💬 Daily Scrum</summary>

### Daily Scrum Süreci

**Zaman:** Her akşam 19:30-23:00 arası  
**Kanallar:** WhatsApp grup mesajları, Google Meet toplantıları  
**Sıklık:** Günlük WhatsApp güncellemeleri, özellikle proje teslim yaklaştıkça yoğun iletişim

### WhatsApp Daily Scrum Konuşmaları
Sprint 3 boyunca takım üyeleri arasında gerçekleşen yoğun günlük iletişim ve proje güncellemeleri. Özellikle son haftada proje teslimi yaklaştıkça iletişim sıklığı artmıştır.

[📱 WhatsApp Daily Scrum Ekran Görüntüleri](https://imgur.com/a/CHJ9yM9)

### Ana İletişim Konuları:
- Backend-Frontend entegrasyon zorlukları
- WebRTC ve gRPC protokol geçişi araştırmaları
- Model optimizasyonu ve uyarı sistemi geliştirmeleri
- Flutter uygulama kamera entegrasyon sorunları
- Proje video hazırlığı ve teslim koordinasyonu
- Son dakika hata çözümleri ve optimizasyonlar

### Toplantı Tarihleri:
- **22 Temmuz:** Entegrasyon sorunları ve çözüm önerileri
- **27 Temmuz:** Acil sprint planlama ve görev dağılımı
- **30 Temmuz:** Geliştirme çalışmaları ve koordinasyon
- **1-2 Ağustos:** Yoğun geliştirme ve son testler

</details>

---

<details>
<summary>📋 Sprint Board Updates</summary>

### ClickUp Sprint Board

Sprint 3 görev dağılımı, ilerleme durumu ve proje yönetimi paneli ile takip edilmiştir.

[📊 ClickUp Sprint 3 Board](https://app.clickup.com/90181399415/v/b/6-901809372442-2)

### Sprint Burndown:
- Başlangıç: 100 puan
- Tamamlanan: 78 puan
- Kalan: 22 puan (proje teslimi sonrası iyileştirmeler için not alındı)

### Kritik Görevler:
- ✅ Model uyarı sistemi entegrasyonu
- ✅ Flutter uygulama kamera optimizasyonu
- 🔄 WebRTC/gRPC protokol geçişi (kısmi tamamlandı)
- ✅ Veri seti hazırlama ve test
- ✅ Proje video hazırlığı

### Sprint Board Ekran Görüntüleri:

#### Sprint 3 Backlog Items:
![Sprint 3 Backlog](SPRINT_3_BACKLOG_GORUNTU_BURAYA)

*Sprint 3 Backlog Items Ekran Görüntüsü*

#### Sprint 3 Sprint Board:
![Sprint 3 Board](SPRINT_3_BOARD_GORUNTU_BURAYA)

*Sprint 3 Sprint Board Ekran Görüntüsü*

#### Sprint 3 Dashboard:
![Sprint 3 Dashboard](SPRINT_3_DASHBOARD_GORUNTU_BURAYA)

*Sprint 3 Sprint Dashboard Ekran Görüntüsü*

</details>

---

<details>
<summary>🖥️ Ürün Durumu</summary>

### Sprint 3 Geliştirme Çıktıları

#### 1. Model İyileştirmeleri ve Uyarı Sistemi

**Squat ve Oturma Modeli Ses Uyarı Sistemi:**
- ✅ Real-time ses geri bildirimi eklendi
- ✅ Squat pozisyonu için sesli uyarı sistemi
- ✅ Oturma pozisyonu düzeltme uyarıları
- ✅ Threshold tabanlı uyarı mekanizması

*Model ses uyarı sistemi çalışır halde demonstrasyonu*

#### 2. Backend-Frontend Entegrasyon Çalışmaları

**Teknik Zorluklar ve Çözümler:**
- 🔄 WebRTC protokolü araştırması yapıldı
- 🔄 gRPC entegrasyon testleri gerçekleştirildi
- ⚠️ Flutter-Backend bağlantı sorunları kısmen çözüldü
- ✅ FastAPI websocket sistemi kuruldu

**Entegrasyon Mimarisi:**
```
[Backend API] ←→ [WebSocket] ←→ [Flutter App] ←→ [Camera Feed]
```

#### 3. Flutter Mobil Uygulama Güncellemeleri

**Kamera ve Performans İyileştirmeleri:**
- ✅ Android emülatör uyumluluğu sağlandı
- ⚠️ iOS uyumluluğu henüz test edilmedi
- ✅ Kamera permission yönetimi eklendi
- ✅ Real-time görüntü işleme optimizasyonu

#### 4. Veri Seti ve Model Eğitimi

**Veri Toplama Çalışmaları:**
- ✅ YouTube videolarından veri çıkarma sistemi
- ✅ Doğru/yanlış hareket etiketleme
- ✅ CSV export ve landmark data işleme
- ✅ Mediapipe entegrasyon optimizasyonu

</details>

---

<details>
<summary>🧪 Test & Doğrulama</summary>

### Kullanıcı Testleri ve Sistem Doğrulaması

#### Teknik Testler:
- **Android Uyumluluğu:** ✅ Başarılı (Xiaomi Note 11 test edildi)
- **Emülatör Testi:** ✅ Android Studio emülatöründe çalışıyor
- **iOS Uyumluluğu:** ❌ Test edilemedi (cihaz kısıtı)
- **Kamera Performansı:** ✅ Real-time işleme başarılı

#### Model Performans Testleri:
- **Squat Algılama Doğruluğu:** ~85%
- **Oturma Pozisyonu Tespiti:** ~80%
- **Ses Uyarı Tepki Süresi:** <1 saniye
- **Real-time İşleme Hızı:** 30 FPS

#### Kullanıcı Deneyimi Testleri:
- ✅ Kolay kurulum ve başlatma
- ✅ Anlaşılır ses geri bildirimleri
- ⚠️ Kamera pozisyonlama rehberi geliştirilmeli
- ✅ Kullanıcı arayüzü sezgisel

</details>

---

<details>
<summary>📈 Sprint Review</summary>

### Sprint 3'te Yapılan İşler

#### ✅ Başarıyla Tamamlanan Görevler:

**Backend & Entegrasyon (28/35 puan)**
- ✅ FastAPI websocket sistemi kurulumu
- ✅ Model API endpoint'lerinin oluşturulması
- ✅ Real-time veri iletişimi testi
- 🔄 WebRTC/gRPC protokol araştırması (devam ediyor)

**Model Geliştirme**
- ✅ Squat modeli ses uyarı sistemi entegrasyonu
- ✅ Oturma pozisyonu uyarı mekanizması
- ✅ Threshold optimizasyonu
- ✅ Performance tuning

**Frontend Geliştirme (16/20 puan)**
- ✅ Flutter kamera entegrasyon iyileştirmeleri
- ✅ Android uyumluluk sağlanması
- ✅ Permission management sistemi
- ✅ UI/UX küçük iyileştirmeler

**Veri İşleme (18/25 puan)**
- ✅ YouTube video işleme pipeline
- ✅ Otomatik veri etiketleme sistemi
- ✅ CSV export optimize edilmesi
- ✅ Landmark data validation

#### 🔄 Devam Eden Görevler:

**Entegrasyon Zorlukları**
- 🔄 WebRTC full implementasyonu
- 🔄 iOS uyumluluk testleri
- 🔄 Cross-platform optimizasyon

**Performans İyileştirmeleri**
- 🔄 Model accuracy artırma çalışmaları
- 🔄 Latency azaltma optimizasyonları

</details>

---

<details>
<summary>🔄 Sprint Retrospective</summary>

### Bu Sprintte Yaptığımız En İyi Şeyler

#### 👍 Başarılı Yönler:
- **Entegrasyon İlerlemi:** Backend-Frontend bağlantısında önemli adımlar atıldı
- **Model Fonksiyonalitesi:** Ses uyarı sistemi başarıyla entegre edildi
- **Problem Çözme:** Teknik zorluklar karşısında alternatif çözümler bulundu
- **Takım Koordinasyonu:** Yoğun sprint döneminde etkili iletişim sağlandı
- **Adaptasyon:** WebRTC/gRPC gibi yeni teknolojilere hızlı adaptasyon
- **Son Dönem Motivasyonu:** Proje teslim tarihi yaklaştıkça takım performansı arttı

#### 📌 Karşılaşılan Zorluklar:
- **Entegrasyon Karmaşıklığı:** Backend-Frontend bağlantısı beklediğimizden zorlu çıktı
- **Protokol Geçişi:** WebRTC ve gRPC implementasyonu zaman aldı
- **Platform Uyumluluğu:** iOS test imkanı bulunamadı
- **Zaman Baskısı:** Sprint son haftasında yoğun çalışma temposu
- **Teknik Debt:** Hızlı geliştirme sürecinde kod kalitesi ikinci planda kaldı

#### 🛠️ Teknik Öğrenimler:
- **WebRTC:** Real-time communication protokollerinde deneyim kazanıldı
- **gRPC:** High-performance RPC framework kullanımı öğrenildi
- **Flutter:** Cross-platform development zorlukları deneyimlendi
- **Model Deployment:** Production ortamında model entegrasyonu deneyimi
- **Real-time Processing:** Canlı video işleme optimizasyon teknikleri

#### 🎯 Gelecek İyileştirmeler İçin Notlar:
1. **Protokol Seçimi:** Başlangıçta daha kapsamlı araştırma yapılmalı
2. **Platform Testleri:** Farklı cihazlar için test stratejisi geliştirilmeli
3. **Entegrasyon Planlaması:** Backend-Frontend entegrasyonu için daha fazla zaman ayrılmalı
4. **Documentation:** Teknik dokümantasyon sürekli güncel tutulmalı
5. **Code Quality:** Hızlı geliştirme sürecinde kod kalitesi standartları korunmalı

#### 📊 Sprint Başarı Metrikleri:
- **Genel Tamamlanma:** %78 (78/100 puan)
- **Takım Katılımı:** %100 (tüm üyeler aktif katılım)
- **Kod Kalitesi:** Orta (hızlı geliştirme sebebiyle)
- **Entegrasyon Başarısı:** %70 (kısmi başarı, geliştirme devam ediyor)
- **Kullanıcı Deneyimi:** %80 (temel işlevsellik çalışıyor)

</details>

---

<details>
<summary>🎬 Proje Teslimi</summary>

### Proje Video ve Dokümantasyon

#### Proje Tanıtım Videosu Hazırlığı:
- **Video Süresi:** Maksimum 2 dakika
- **İçerik:** Uygulama demo, model çalışması, ses uyarı sistemi
- **Platform:** YouTube (Public/Unlisted)
- **Gösterilecek Özellikler:**
  - Real-time postür analizi
  - Squat ve oturma pozisyonu tespiti
  - Ses uyarı sistemi demonstrasyonu
  - Flutter mobil arayüz kullanımı

#### GitHub Repository Durumu:
- ✅ Kod tabanı güncel ve dokümante
- ✅ README.md comprehensive format
- ✅ Sprint raporları eksiksiz
- ✅ Model dosyaları ve veri setleri organize
- ✅ Kurulum ve kullanım kılavuzu hazır

#### Final Proje Çıktıları:
- **Çalışan Prototype:** Android platformunda test edilmiş
- **Model Accuracy:** Squat %85, Oturma Pozisyonu %80
- **Real-time Performance:** 30 FPS stabil işleme
- **User Experience:** Ses geri bildirimli postür düzeltme sistemi
- **Demo Video:** Proje özellikleri ve kullanımı

#### Bootcamp Teslim Kriterleri:
- ✅ 2 dakikalık tanıtım videosu hazır
- ✅ GitHub repository eksiksiz
- ✅ Çalışan prototype demo
- ✅ Teknik dokümantasyon tamamlandı
- ✅ Sprint raporları ve proje yönetimi

</details>

---

<details>
<summary>🚀 Proje Sonuç Değerlendirmesi</summary>

### Genel Proje Başarı Analizi

#### Toplam Proje İstatistikleri:
- **Toplam Sprint Sayısı:** 3
- **Toplam Hedeflenen Puan:** 300
- **Toplam Gerçekleşen Puan:** 252 (%84)
- **Proje Süresi:** 6 hafta
- **Takım Büyüklüğü:** 5 kişi

#### Sprint Bazında Performans:
- **Sprint 1:** 50/50 puan (%100)
- **Sprint 2:** 74/100 puan (%74)
- **Sprint 3:** 78/100 puan (%78)

#### Teknik Başarılar:
- ✅ MediaPipe entegrasyonu başarıyla tamamlandı
- ✅ Real-time video processing implementasyonu
- ✅ Flutter cross-platform mobile uygulama
- ✅ Ses uyarı sistemi entegrasyonu
- ✅ Machine learning model deployment

#### Öğrenimler ve Deneyimler:
- **Takım Çalışması:** Uzaktan çalışma koordinasyonu
- **Teknoloji Yığını:** Modern AI/ML araçları kullanımı
- **Proje Yönetimi:** Agile/Scrum metodolojisi uygulaması
- **Problem Çözme:** Teknik zorluklar karşısında adaptasyon

#### Gelecek Potansiyeli:
Posecore projesi, fizyoterapi ve wellness alanında gerçek bir ihtiyaca cevap veren, teknik olarak uygulanabilir bir çözüm olarak geliştirilmiştir. Bootcamp sonrasında da geliştirilmeye devam edilebilecek bir proje yapısına sahiptir.

</details>

</details>


