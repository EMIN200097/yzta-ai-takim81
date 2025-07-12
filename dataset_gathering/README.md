
# Veri Seti Toplama

Bu dizin, projede kullanılacak veri setlerini toplamak, işlemek ve düzenlemek için kullanılan araçları içerir. Ham video verisini Makine Öğrenmesi modellerine beslemek ve doğru postür analizi yapmak için kullanılacak bu veriyi toplamayı burada gerçekleştiriyoruz.

## İçerik

### main.py ile Video'dan CSV'ye Dönüştürme

`main.py`, `input/` klasöründeki bir video dosyasını alır ve işlenmiş verileri `output/` klasöründe bir CSV dosyasına kaydeder. Temel adımlar:

1. `input/` klasörüne dönüştürmek istediğiniz video dosyasını ekleyin.
2. `main.py:173`de video_name argümanına string olarak doğru oturuş postürü videosu dosyanızın ismini **dosya tipi uzantısıyla beraber** verin. -ya da kamera kullanmak için bu keyword argümanı silin.-
3. Programı çalıştırıp mevcut kareleri kaydetmeye başlamak için klavyenizdeki "L" tuşuna basın.
4. Program, videodaki kareleri işler ve ilgili verileri `output/` dizinindeki CSV dosyasına yazar. Yazmayı durdurmak için tekrardan "L" tuşuna basabilirsiniz.
5. Programdan çıkış yapmak için klavyenizdeki "Q" tuşuna basın.

Detaylı parametreler ve ek seçenekler için `main.py` dosyasındaki açıklamaları inceleyin.

## Notlar

- Bir dizin yukarıdaki requirements.txt dosyasındaki gereksinimleri pip ile kurduğunuzdan emin olun.
- Tam olarak akışa hakim olmak için main.py dosyasındaki komut satırlarını okuyun
- CSV dosyası kullanılmadan önce gözden geçirilmelidir.
