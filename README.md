# 👤 Unsupervised Face Clustering on News Photos

This pipeline finds the **10 most frequently appearing people** in a university's news photos **without any labeled data or name database**. It scrapes the images, detects faces, turns each face into an embedding with a deep CNN, and groups identical people with **DBSCAN** clustering.

Final project for the *Visual Programming / Image Processing* course at Balıkesir University (Spring 2026).

## How It Works

```
balikesir.edu.tr/haberler
        │  1. Scrape (requests + BeautifulSoup, paginated, filtered by month)
        ▼
   news images
        │  2. Detect faces (MTCNN, confidence ≥ 0.80, min size 30 px)
        ▼
   face crops
        │  3. Embed (InceptionResnetV1 pretrained on VGGFace2 → 512-d vectors)
        ▼
   embeddings
        │  4. Cluster (DBSCAN, cosine distance, eps = 0.20, min_samples = 2)
        ▼
   identity clusters → rank by size → save the top 10 faces
```

1. **Scraping:** walks the news listing page by page, opens each article, parses its date and collects only the photos from the target month (May 2026). It stops as soon as it reaches older news.
2. **Face detection:** uses MTCNN from `facenet-pytorch`. Low-confidence and very small detections are discarded.
3. **Embeddings:** each face is mapped to a 512-dimensional vector in which faces of the same person lie close together.
4. **Clustering:** DBSCAN groups the embeddings without needing the number of people in advance. Noise points (label `-1`) are dropped. A strict `eps` prevents one giant cluster from forming.
5. **Report:** prints the top 10 clusters by size and saves one representative crop per person to `top_10_faces/`.

The pipeline runs on the GPU (CUDA) when one is available and falls back to the CPU otherwise.

## Tech Stack

| Step | Tools |
|---|---|
| Scraping | requests, BeautifulSoup4 |
| Detection and embedding | PyTorch, facenet-pytorch (MTCNN, InceptionResnetV1 / VGGFace2) |
| Clustering | scikit-learn (DBSCAN) |
| Image I/O | OpenCV, Pillow, NumPy |

## Getting Started

```bash
git clone https://github.com/AlperSayginEtci/Goruntu_Isleme_Finel_Odevi.git
cd Goruntu_Isleme_Finel_Odevi
pip install -r requirements.txt
python main.py
```

On the first run, the pretrained VGGFace2 weights are downloaded automatically. The output is written to:
- `images/`: downloaded news photos
- `top_10_faces/`: one cropped face per top-10 person, named like `Sira_1_KisiID_4_23_Kez.jpg` (rank, cluster ID, count)

## Notes
- Everything runs locally. No names or identities are attached to the clusters.
- `eps` is the key hyperparameter. A higher value merges different people, and a lower value splits one person into several clusters.
