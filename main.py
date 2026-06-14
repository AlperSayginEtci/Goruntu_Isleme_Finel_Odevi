import os
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.cluster import DBSCAN
from collections import Counter
import shutil
import urllib3
from tqdm import tqdm
# SSL uyarılarını gizle
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Klasör ayarları
IMAGES_DIR = "images"
TOP_10_DIR = "top_10_faces"

def setup_directories():
    for d in [IMAGES_DIR, TOP_10_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

def download_images_from_news(url="https://balikesir.edu.tr/haberler"):
    """
    Sitedeki haberleri sayfalayarak (pagination) tarayıp sadece Mayıs 2026
    ayına ait olan haberlerin içindeki fotoğrafları indirir.
    """
    print(f"[*] {url} adresi taranıyor ve Mayıs 2026 haber fotoğrafları aranıyor...")
    setup_directories()
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    months = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]
    img_urls = []
    
    page = 1
    stop_scraping = False
    
    while not stop_scraping:
        page_url = f"{url}?page={page}"
        print(f"  -> Sayfa {page} taranıyor...")
        try:
            response = requests.get(page_url, headers=headers, verify=False, timeout=15)
            if response.status_code != 200:
                break
                
            soup = BeautifulSoup(response.text, "html.parser")
            news_links = soup.find_all('a', class_='northwood-news__news-item-content')
            
            if not news_links:
                break # Sayfa sonuna gelinmiş olabilir
                
            for link in news_links:
                href = link.get('href')
                if not href: continue
                
                if href.startswith('haberler/'):
                    article_url = "https://balikesir.edu.tr/" + href
                elif href.startswith('http'):
                    article_url = href
                else:
                    article_url = "https://balikesir.edu.tr/haberler/" + href
                
                try:
                    art_resp = requests.get(article_url, headers=headers, verify=False, timeout=15)
                    art_soup = BeautifulSoup(art_resp.text, "html.parser")
                    
                    # Tarih metnini bul
                    date_found = ""
                    for tag in art_soup.find_all(['span', 'p', 'div', 'time']):
                        text = tag.get_text(strip=True).lower()
                        if '2026' in text and any(m in text for m in months) and len(text) < 50:
                            date_found = text
                            break
                    
                    if date_found:
                        # Eğer tarih Mayıs 2026 ise resimleri topla
                        if 'mayıs' in date_found or '05.2026' in date_found or '05/2026' in date_found:
                            content_div = art_soup.find('div', class_='entry-content')
                            if content_div:
                                imgs = content_div.find_all('img')
                                for img in imgs:
                                    src = img.get('src')
                                    if src and '/uploads/' in src and 'logo' not in src.lower():
                                        if not src.startswith("http"):
                                            src = "https://balikesir.edu.tr" + src if src.startswith("/") else "https://balikesir.edu.tr/" + src
                                        img_urls.append(src)
                        
                        # Eğer tarih Nisan 2026, Mart 2026 vb. geçmişse aramayı durdur.
                        earlier_months = ["ocak", "şubat", "mart", "nisan"]
                        if any(m in date_found for m in earlier_months) and '2026' in date_found:
                            print(f"  [!] '{date_found}' bulundu. Mayıs 2026'dan eski bir habere ulaşıldığı için tarama durduruluyor.")
                            stop_scraping = True
                            break
                        
                        if '2025' in date_found:
                            print(f"  [!] 2025 yılına ait haber bulundu. Tarama durduruluyor.")
                            stop_scraping = True
                            break
                            
                except Exception as e:
                    pass
            
            if stop_scraping:
                break
                
            page += 1
            if page > 50: # Her ihtimale karşı güvenlik sınırı
                break
                
        except Exception as e:
            print(f"[!] Sayfa tarama hatası: {e}")
            break
            
    # Eğer Mayıs 2026 haberi bulunamadıysa fallback (Güvenlik)
    if not img_urls:
        print("[!] Taranan sayfalarda 'Mayıs 2026' ayına ait uygun haber resmi bulunamadı.")
        print("[*] Proje algoritmasını test etmek için ilk sayfadaki genel güncel haber resimleri çekiliyor...")
        try:
            response = requests.get(url, headers=headers, verify=False, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            for img in soup.find_all("img"):
                src = img.get("src")
                if src and '/uploads/' in src:
                    if not src.startswith("http"):
                        src = "https://balikesir.edu.tr" + src if src.startswith("/") else "https://balikesir.edu.tr/" + src
                    img_urls.append(src)
        except:
            pass

    # Çift linkleri temizle
    img_urls = list(dict.fromkeys(img_urls))
    print(f"[*] Toplam {len(img_urls)} adet uygun resim bulundu. İndiriliyor...")
    
    count = 0
    for i, img_url in enumerate(img_urls):
        try:
            img_data = requests.get(img_url, headers=headers, verify=False, timeout=10).content
            if len(img_data) > 5000:
                with open(os.path.join(IMAGES_DIR, f"image_{i}.jpg"), 'wb') as handler:
                    handler.write(img_data)
                count += 1
        except Exception:
            pass
    print(f"[*] {count} adet resim '{IMAGES_DIR}' klasörüne indirildi.\n")

def process_and_cluster_faces():
    """
    İndirilen resimlerdeki yüzleri bulur ve herhangi bir isim 
    veritabanı kullanmadan DBSCAN ile gruplar.
    """
    print("[*] İndirilen resimlerdeki yüzler tespit ediliyor (Öznitelik/Embedding Çıkarımı)...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] PyTorch başlatıldı. Kullanılan donanım: {device.type.upper()}")
    
    # MTCNN yüz tespit modeli ve InceptionResnetV1 embedding modeli
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=30)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    encodings = []
    face_crops = []
    
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    for filename in tqdm(image_files, desc="Yüzler Taranıyor"):
        img_path = os.path.join(IMAGES_DIR, filename)
        try:
            img_pil = Image.open(img_path).convert('RGB')
            
            # Yüzlerin yerlerini bul ve kırpılmış tensor hallerini al
            boxes, probs = mtcnn.detect(img_pil)
            
            if boxes is not None:
                faces_tensor = mtcnn(img_pil)
                if faces_tensor is not None:
                    # Modelden geçirerek 512 boyutlu vektörleri çıkar (GPU üzerinde)
                    embeddings = resnet(faces_tensor.to(device)).detach().cpu().numpy()
                    
                    img_cv = cv2.imread(img_path)
                    
                    for i, box in enumerate(boxes):
                        if probs[i] is None or probs[i] < 0.80:
                            continue
                            
                        x1, y1, x2, y2 = [int(b) for b in box]
                        x1, y1 = max(0, x1), max(0, y1)
                        face_crop = img_cv[y1:y2, x1:x2]
                        
                        if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                            continue
                            
                        encodings.append(embeddings[i])
                        face_crops.append(face_crop)
                        
        except Exception:
            pass

    if not encodings:
        print("[!] Hiç insan yüzü bulunamadı.")
        return

    print(f"\n[*] Toplam {len(encodings)} adet insan yüzü tespit edildi.\n")
    
    # --- İSİM KULLANMADAN YÜZ EŞLEŞTİRME (DBSCAN) ---
    print("[*] İsim veritabanı kullanılmadan DBSCAN algoritması ile yüzler kümeleniyor...")
    X = np.array(encodings)
    
    # eps=0.20: facenet-pytorch modelinin cosine metric'ine göre daha katı bir sınır. (Giant cluster oluşumunu engeller)
    # min_samples=2: Yanlış yüzleri (noise) azaltmak için en az 2 defa görünenleri grupla.
    clt = DBSCAN(eps=0.20, metric="cosine", min_samples=2)
    clt.fit(X)
    
    labels = clt.labels_
    counts = Counter(labels)
        
    # -1 etiketli olanlar (noise/gürültü/hiçbir gruba girmeyenler) listeye dahil edilmemeli
    if -1 in counts:
        del counts[-1]
        
    # En sık görünen 10 yüzü raporla ve sayılarını yazdır
    top_10 = counts.most_common(10)
    
    print("\n=======================================================")
    print("      EN SIK GÖRÜNEN İLK 10 YÜZ RAPORU (İSİMSİZ)       ")
    print("=======================================================")
        
    for rank, (person_id, count) in enumerate(top_10, 1):
        print(f"{rank}. Sırada: (Kişi ID: {person_id}) => Toplam {count} defa tespit edildi.")
        
        person_indices = np.where(labels == person_id)[0]
        first_face_img = face_crops[person_indices[0]]
        
        out_path = os.path.join(TOP_10_DIR, f"Sira_{rank}_KisiID_{person_id}_{count}_Kez.jpg")
        cv2.imwrite(out_path, first_face_img)

    print("\n=======================================================")
    print(f"[*] İşlem tamamlandı. En çok görülen ilk 10 kişinin kırpılmış yüzleri '{TOP_10_DIR}' klasörüne kaydedildi.")

if __name__ == "__main__":
    print("=======================================================")
    print("   GÖRSEL PROGRAMLAMA FİNAL PROJESİ - YÜZ KÜMELEME")
    print("=======================================================")
    download_images_from_news("https://balikesir.edu.tr/haberler")
    process_and_cluster_faces()
