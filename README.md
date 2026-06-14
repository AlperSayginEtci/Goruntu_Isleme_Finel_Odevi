# Görsel Programlama Final Projesi - Yüz Kümeleme (Face Clustering)

Bu proje, Balıkesir Üniversitesi haber sayfasından (`balikesir.edu.tr/haberler`) fotoğrafları otomatik indirerek, önceden hazırlanmış bir isim veri tabanı kullanmadan gözetimsiz öğrenme (DBSCAN) yöntemiyle en sık görünen ilk 10 kişiyi tespit etmeyi amaçlar.

## Proje Yapısı
- `main.py`: Projenin (Web Scraping, Yüz Tespiti, Kümeleme ve Raporlama) tamamını çalıştıran ana kod.
- `requirements.txt`: Projenin çalışması için gereken Python kütüphaneleri listesi.
- `images/` (Çalıştırılınca oluşur): Siteden indirilen haber fotoğrafları.
- `top_10_faces/` (Çalıştırılınca oluşur): Sistem tarafından en sık bulunduğu tespit edilen 10 kişinin ilk kırpılmış yüzleri.

## Kurulum ve Çalıştırma (Teams Teslimi Öncesi)

**1. Kütüphanelerin Yüklenmesi:**
Proje dizininde bir terminal (komut satırı) açın ve aşağıdaki kodu yazarak gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```
*(Not: `torch`, `facenet-pytorch`, `opencv-python` ve `scikit-learn` arka planda gerekli makine öğrenmesi araçlarını kuracaktır. İnternet hızınıza ve PyTorch boyutlarına bağlı olarak birkaç dakika sürebilir.)*

**2. Projeyi Çalıştırma:**
```bash
python main.py
```
Program çalıştığında sırasıyla:
1. Web sitesine bağlanıp `images` klasörüne resimleri indirecek.
2. Makine öğrenmesi modeli (MTCNN ve InceptionResnetV1) ile her bir yüzün 512 boyutlu vektörünü çıkaracak (ilk çalışmada VGGFace2 modelinin küçük bir dosyasını internetten otomatik indirebilir, izin verin).
3. **DBSCAN** algoritması ile bu vektörlerin birbirine uzaklıklarını ölçerek aynı kişileri aynı "Küme ID"si altına toplayacak. (İsim kullanmadan eşleştirme mantığı burasıdır).
4. Konsolda 1'den 10'a kadar bir sıralama yapıp, bu kişilerin kırpılmış yüz fotoğraflarını `top_10_faces` klasörüne kaydedecektir.

## Teams'e Nasıl Yüklemelisiniz?
Projenin çalıştığından emin olduktan sonra:
1. Oluşan `images/` ve `top_10_faces/` klasörlerinin içini boşaltabilirsiniz (Hoca kendisi çalıştırıp sonuçları görecektir).
2. `main.py`, `requirements.txt`, `README.md` ve daha önce oluşturulan `proje_cozum_yontemi.md` dosyasını seçip **sağ tıklayarak ZIP haline (Arşive Ekle)** getirin.
3. Oluşan bu ZIP dosyasını Teams'e Final Sınavı ödevi olarak yükleyebilirsiniz.

Başarılar dilerim!
