---
title: "Cosine Similarity"
date: "2025-07-07"
---

## Giới thiệu: Khi máy tính cần “đo” sự tương đồng

Trong thực tế, khi bạn tìm kiếm phim trên Netflix hoặc gợi bên Google: làm sao máy tính biết hai đoạn văn hay hai câu văn có liên quan với nhau?

Ví dụ:
- "Tôi thích phim hành động"
- "Tôi mê thể loại phim action"

Con người dễ thấy sự liên quan. Máy tính làm được như vậy nhờ vào việc **biểu diễn văn bản dưới dạng vector** và **tính độ tương đồng bằng Cosine Similarity**.

---

## Cosine Similarity là gì?

Cosine similarity là một metric đo **góc giữa hai vector trong không gian**:

$$
\text{cs}(\vec{x}, \vec{y}) = \frac{\vec{x} \cdot \vec{y}}{\|\vec{x}\| \cdot \|\vec{y}\|} = \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2} \cdot \sqrt{\sum_{i=1}^n y_i^2}}
$$

**Ý nghĩa:**
- Cosine = 1 ⇒ cùng hướng, giống nhau hoàn toàn
- Cosine = 0 ⇒ vuông góc, không liên quan
- Cosine = -1 ⇒ đối hướng


**Điểm hay của cosine:** 
- Bỏ qua độ dài vector  
Nếu nhân cả hai vector với hằng số dương bất kỳ, cosine similarity vẫn giữ nguyên:

$$
\text{cs}(a \vec{x}, b \vec{y}) = \text{cs}(\vec{x}, \vec{y}) \quad \text{với } ab > 0
$$

**Chứng minh:**

$$
\text{cs}(a \vec{x}, b \vec{y}) = \frac{a \vec{x} \cdot b \vec{y}}{\|a \vec{x}\| \cdot \|b \vec{y}\|}
= \frac{ab \sum x_i y_i}{\sqrt{a^2 \sum x_i^2} \cdot \sqrt{b^2 \sum y_i^2}}
= \text{cs}(\vec{x}, \vec{y})
$$
![No Dependence Length](/images/M02W01/NoDependenceLength.png)
- Tập trung vào hướng – lý tưởng cho NLP

---

## So sánh 3 cách cài đặt Cosine Similarity trong Python

### 1. Dùng `numpy` (hiệu quả, phổ biến)

```python
import numpy as np

def cosine_similarity_np(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Test
vector1 = [5, 3, 2, 7]
vector2 = [2, 9, 4, 1]
print("Numpy:", cosine_similarity_np(vector1, vector2))

```
--- 

### 2.Dùng sklearn (chuẩn hóa sẵn, dễ dùng cho nhiều vector)
```python 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vector1 = np.array([[5, 3, 2, 7]])
vector2 = np.array([[2, 9, 4, 1]])
print("Sklearn:", cosine_similarity(vector1, vector2)[0][0])
```
---

### 3.Dùng scipy (hữu ích với sparse vector như TF-IDF)
```python 
from scipy.spatial.distance import cosine

vector1 = [5, 3, 2, 7]
vector2 = [2, 9, 4, 1]
print("Scipy:", 1 - cosine(vector1, vector2))
```

---

## TF-IDF: Biến văn bản thành vector

Trước khi tính độ tương đồng giữa hai đoạn văn bằng **cosine similarity**, ta cần **chuyển văn bản thành vector số**. Một trong những cách cổ điển nhất là dùng **TF-IDF**.

---

### TF-IDF là gì?

**TF-IDF** là viết tắt của:

- **TF (Term Frequency)** – Tần suất xuất hiện của một từ trong một văn bản.
- **IDF (Inverse Document Frequency)** – Mức độ hiếm của từ trong toàn bộ tập văn bản.

Công thức chuẩn:

$$
\text{TFIDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{1 + \text{DF}(t)}\right)
$$

- \( t \): một từ (term)
- \( d \): một tài liệu cụ thể
- \( N \): tổng số tài liệu
- \( DF(t) \): số tài liệu chứa từ \( t \)

---

### Giải thích ý nghĩa

- **TF**: Từ xuất hiện càng nhiều trong một văn bản ⇒ càng quan trọng ⇒ điểm TF cao.
- **IDF**: Từ xuất hiện quá phổ biến (ví dụ: "là", "và", "the", "is") ⇒ không mang nhiều thông tin ⇒ nên giảm trọng số bằng cách nhân với IDF thấp.

⟶ Như vậy, TF-IDF **ưu tiên các từ đặc trưng và quan trọng** cho văn bản, bỏ qua các từ phổ biến không mang ý nghĩa so sánh.

---

### Minh hoạ: Tính cosine giữa hai câu

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "Tôi thích phim hành động",
    "Tôi mê thể loại phim action"
]

vec = TfidfVectorizer()
tfidf = vec.fit_transform(docs)
score = cosine_similarity(tfidf[0], tfidf[1])

print("Độ tương đồng:", score[0][0])
```

Kết quả: Điểm cosine cao ~ 0.85 cho thấy hai câu rất liên quan.

---
## Vì sao IDF dùng log? Và tại sao cộng thêm 1?

Khi tính **TF-IDF**, phần **IDF** có vai trò giúp giảm trọng số của những từ quá phổ biến trong tập dữ liệu.

Công thức IDF thường thấy:

$$
\text{IDF}(t) = \log\left(\frac{N}{1 + DF(t)}\right)
$$

### Giải thích từng phần:

-  DF(t): là số văn bản chứa từ  t 
-  N : là tổng số văn bản
- **Thêm 1 vào mẫu số**: giúp **tránh chia cho 0** trong trường hợp từ đó chưa từng xuất hiện (hiếm nhưng vẫn có thể xảy ra trong quá trình khởi tạo hoặc inference).

---

###  Vì sao dùng hàm `log()`?

1. **Đánh phạt từ phổ biến** nhưng không quá tay.

   - Nếu dùng 
   $$ 
   \frac{N}{DF(t)} 
   $$ 
   trực tiếp, ta sẽ **phạt quá nặng** các từ phổ biến.
   - Dùng `log` giúp **làm mượt lại sự chênh lệch**, giữ tỉ lệ hợp lý giữa các từ.

2. **Hàm log tăng chậm** – điều này giúp giữ thăng bằng giữa "từ hiếm" và "từ vừa phổ biến".

 **Từ phổ biến có IDF ≈ 0**, còn từ hiếm có IDF cao ⇒ tạo sự phân biệt rõ nét.
![Log function graph](/images/M02W01/log.png)
---

### Vì sao cộng thêm 1?

Cộng thêm 1 trong mẫu số:

$$
\log\left(\frac{N}{1 + DF(t)}\right)
$$

- Tránh trường hợp chia cho 0 khi  DF(t) = 0 
- Đảm bảo **IDF luôn có giá trị hợp lệ**
- Đặc biệt hữu ích khi làm inference với từ chưa gặp trong training data

---

## Hạn chế của TF-IDF + Cosine

| Vấn đề | Giải thích |
|-----------|-------------|
| Không hiểu ngữ nghĩa | "bạn yêu tôi" và "tôi yêu bạn" ra vector khác |
| Nhạy với từ đồng nghĩa | "phim" vs "movie" |
| Sparse vector | Rất nhiều chiều |

---

## Ta cần việc biến từ ngữ thành vector hiểu ngữ nghĩa hơn TF-IDF

Hiện nay, TF-IDF thường được thay thế bằng **embedding** như:

- Word2Vec, GloVe: vector từ
- SBERT, USE: vector cả câu

Nhưng cosine similarity vẫn được giữ lại là metric chính để **so sánh** vector văn bản.

**Điểm mạnh:**
- Dễ hiểu, dễ tính
- Phù hợp cho vector sparse lẫn dense

---

## Ứng dụng: Hệ thống gợi ý phim dựa trên tìm kiếm nội dung

### 1. Bối cảnh và lý do thực hiện

Trong bối cảnh ngành công nghiệp giải trí đang bùng nổ, số lượng phim mới được sản xuất và công bố tăng mạnh mỗi năm. Điều này tạo ra một thách thức lớn cho người dùng: **quá tải thông tin**. Với hàng ngàn lựa chọn, việc tìm ra một bộ phim phù hợp với sở thích cá nhân trở nên mất thời gian và kém hiệu quả.

Hiện nay, nhiều nền tảng đề xuất phim (movie recommender) đã tồn tại, nhưng chúng thường:
- Yêu cầu người dùng trả phí để sử dụng đầy đủ tính năng
- Không đáp ứng tốt nhu cầu cá nhân hóa theo nội dung tìm kiếm

Chính vì vậy, chúng tôi xây dựng một **hệ thống gợi ý phim dựa trên tìm kiếm từ khóa nội dung** – ứng dụng các kỹ thuật học trong môn Information Retrieval, đặc biệt là mô hình **TF-IDF và cosine similarity**.

---

### 2. Mô tả hệ thống

#### Mục tiêu:
- Cho phép người dùng nhập **từ khóa tự do** (ví dụ: `"batman joker"`), hệ thống sẽ trả về danh sách các bộ phim liên quan nội dung theo mức độ tương đồng cao.
- Hỗ trợ lọc kết quả theo **thể loại**, **đánh giá**, và các yếu tố cá nhân hóa khác.

#### Các thành phần chính:
- **Tiền xử lý dữ liệu**: Loại bỏ stopwords, token hóa, chuẩn hóa từ (stemming).
```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os, json

english_stopwords = set(stopwords.words("english"))

corpus = []
doc_ids = []

for filename in os.listdir("Data"):
    with open(os.path.join("Data", filename), "r", encoding="utf-8") as file:
        data = json.load(file)
        text = data["Overview"]
        tokens = word_tokenize(text.lower())
        tokens = [PorterStemmer().stem(t) for t in tokens if t not in english_stopwords]
        corpus.append(" ".join(tokens))
        doc_ids.append(filename)
```

- **Xây dựng chỉ mục TF-IDF**: Sử dụng `TfidfVectorizer` từ thư viện `scikit-learn`.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import json

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

index = {}
for token, idx in vectorizer.vocabulary_.items():
    index[token] = {doc_ids[i]: tfidf_matrix[i, idx] for i in range(len(doc_ids))}

with open("indexUpdate.json", "w") as f:
    json.dump(index, f)
```

- **Truy vấn và xếp hạng**: So sánh vector truy vấn với tập phim bằng **cosine similarity**, sau đó **xếp hạng top K kết quả**.
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words("english"))

def compute_df(index):
    return {term: sum(1 for score in docs.values() if score > 0) for term, docs in index.items()}

def compute_tf_idf(term, query_tokens, index, df):
    N = len(set(doc_id for docs in index.values() for doc_id in docs))
    tf = query_tokens.count(term) / len(query_tokens)
    idf = np.log(N / (1 + df[term]))
    return tf * idf

def search(query, index_file="indexUpdate.json", top_k=12):
    with open(index_file, "r", encoding="utf-8") as file:
        index = json.load(file)
    tokens = [PorterStemmer().stem(t) for t in word_tokenize(query.lower()) if t not in english_stopwords]
    vocabulary = list(index.keys())
    df = compute_df(index)

    query_vector = {t: compute_tf_idf(t, tokens, index, df) for t in tokens if t in vocabulary}

    doc_vectors = {}
    for token, docs in index.items():
        for doc_id, score in docs.items():
            doc_vectors.setdefault(doc_id, {})[token] = score

    query_vec = np.array([query_vector.get(t, 0) for t in vocabulary]).reshape(1, -1)
    doc_matrix = np.array([[doc_vectors[d].get(t, 0) for t in vocabulary] for d in doc_vectors])

    similarities = cosine_similarity(query_vec, doc_matrix).flatten()
    results = sorted(zip(doc_vectors.keys(), similarities), key=lambda x: -x[1])[:top_k]
    return results
```
- **Giao diện người dùng (UI)**: Triển khai với `Streamlit` – mỗi kết quả có hình ảnh, tiêu đề, mô tả ngắn; cho phép lọc theo sở thích (thể loại, rating,...).

---

### 3. Quy trình hoạt động

1. **Người dùng nhập truy vấn** (ví dụ: `"batman joker"`).
2. Truy vấn được xử lý như một đoạn văn bản:
   - Tokenize, loại stopwords, và vector hóa bằng TF-IDF.
3. So sánh truy vấn với tất cả các phim trong kho dữ liệu:
   - Tính **cosine similarity** giữa vector truy vấn và từng vector phim.
4. Sắp xếp kết quả theo điểm tương đồng, trả về **top K phim liên quan nhất**.
5. Người dùng có thể lọc kết quả theo **thể loại (genre), điểm đánh giá (rating), năm phát hành**,...

---

### 4. Ví dụ minh hoạ

![demo](/images/M02W01/demo.png)
cụ thể hơn mọi người có thể xem code tại repo : https://github.com/nmquan22/MovieSeachFromText

---
## Kết lại

- Cosine similarity là **viên gạch nối** giữa vector văn bản và khả năng so sánh ngữ nghĩa
- TF-IDF là bước đầu, embedding là bước tiếp theo
- Cosine similarity vẫn là metric được tin dùng cho semantic search

---

## Tham khảo

- [MachineLearningPlus - Cosine Similarity](https://www.machinelearningplus.com/nlp/cosine-similarity)
- [Comparing Text Embeddings: TF-IDF vs SBERT](https://medium.com/@venugopal.adep/comparative-study-of-text-embeddings-tf-idf-vs-sentence-transformer-28627c315f21)
- [Cosine Similarity Video Explanation](https://www.youtube.com/watch?v=e9U0QAFbfLI)
- [Universal Sentence Encoder Paper](https://arxiv.org/abs/1803.11175)
