---
title: 'Phân loại Naive Bayes và Ứng dụng trong NLP'
date: '2025-07-14'
---
## Định lý Bayes là gì?

Định lý Bayes là một công cụ toán học cực kỳ quan trọng trong xác suất thống kê, cho phép ta **tính xác suất của một giả thuyết dựa trên thông tin quan sát được**.

### Bắt đầu từ xác suất đồng thời

Giả sử A và B là hai biến cố bất kỳ, ta có hai cách diễn tả xác suất đồng thời P(A, B):  
Giả sử ta quan tâm đến việc chọn hai biến cố A và B theo thứ tự:

1. Đầu tiên chọn A: xác suất xảy ra là P(A).  
2. Sau khi biết A xảy ra, tiếp tục chọn B: xác suất là P(B|A).

Do đó xác suất xảy ra cả A lẫn B (theo thứ tự) là tích:

$$
P(A, B) = P(A) \cdot P(B|A)
$$

Ngược lại, nếu chọn B trước rồi A thì:

$$
P(A, B) = P(B) \cdot P(A|B)
$$

=> Kết quả cuối cùng là:

$$
P(A, B) = P(A) \cdot P(B|A) = P(B) \cdot P(A|B)
$$
Từ đây, ta có thể suy ra **Định lý Bayes** bằng cách biến đổi công thức:

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

### Ý nghĩa các thành phần:

- $P(A|B)$: Xác suất hậu nghiệm — xác suất A xảy ra khi đã biết B  
- $P(B|A)$: Xác suất có điều kiện — xác suất B xảy ra khi biết A  
- $P(A)$: Xác suất tiên nghiệm của A  
- $P(B)$: Xác suất toàn phần — độ tin cậy của thông tin quan sát B

---
## Biểu diễn văn bản bằng Bag of Words

Trong các bài toán phân loại văn bản, dữ liệu đầu vào thường là **chuỗi ký tự** (câu, đoạn văn). Để áp dụng mô hình Naive Bayes, ta cần chuyển văn bản thành dạng **số học**, cụ thể là một vector đặc trưng.

Một kỹ thuật phổ biến là **Bag of Words (BoW)**, với ý tưởng:

- Mỗi văn bản được biểu diễn bằng một vector chứa số lần xuất hiện của từng từ trong từ điển (vocabulary).
- Không quan tâm đến vị trí của từ, chỉ đếm tần suất.

### Ví dụ đơn giản:

Giả sử có 3 văn bản:

1. "Tôi thích học máy"  
2. "Máy học rất thú vị"  
3. "Tôi không thích spam"

Từ điển tổng hợp (vocabulary):  
`["tôi", "thích", "học", "máy", "rất", "thú", "vị", "không", "spam"]`

Biểu diễn văn bản 1 dạng BoW vector:  
`[1, 1, 1, 1, 0, 0, 0, 0, 0]`  
(1 lần "tôi", 1 lần "thích", 1 lần "học", 1 lần "máy", các từ khác không có)

Mỗi dòng là một vector \(X = (x_1, x_2, ..., x_n)\) — dùng làm đầu vào cho mô hình Naive Bayes.

> Ghi chú: Các kỹ thuật nâng cao hơn như TF-IDF hay word embeddings có thể thay thế BoW, nhưng BoW vẫn là lựa chọn mặc định trong nhiều hệ thống Naive Bayes đơn giản.
![Image BOW](/images/M02W02/boW.png)
---


## Ứng dụng trong Mô hình Naive Bayes

**Naive Bayes** là một mô hình học máy dựa trực tiếp trên định lý Bayes, áp dụng vào bài toán phân loại.

Giả sử:

- $C$: là một lớp (label), ví dụ: spam / không spam  
- $X = (x_1, x_2, ..., x_n)$: là các đặc trưng đầu vào (ví dụ: các từ trong văn bản)

Ta cần tính:

$$
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
$$

---

### Giả định “naive” – Độc lập có điều kiện

Naive Bayes giả định rằng các đặc trưng $x_i$ là độc lập với nhau khi đã biết lớp $C$, tức là:

$$
P(X|C) = P(x_1, x_2, ..., x_n | C) = \prod_{i=1}^{n} P(x_i | C)
$$

→ Khi đó, công thức Bayes được đơn giản hóa:

$$
P(C|X) \propto P(C) \cdot \prod_{i=1}^{n} P(x_i | C)
$$

> Dấu $\propto$ có nghĩa là **tỉ lệ thuận** – ta có thể bỏ qua mẫu số $P(X)$ vì nó không phụ thuộc vào lớp $C$.

---
## Vì sao tính $P(x_i \mid C)$ dễ hơn $P(C \mid X)$?

Khi áp dụng định lý Bayes để phân loại văn bản, mục tiêu của ta là tính:

$$
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
$$

Tuy nhiên, tính trực tiếp $P(C|X)$ là rất khó, vì:

- $X$ có thể là một văn bản dài, tổ hợp của nhiều từ khác nhau
- Không có cách nào để biết chắc xác suất một câu xuất hiện trong lớp cụ thể nếu không phân tích từng phần tử trong nó

### Giải pháp của Naive Bayes

Naive Bayes "đảo ngược" vấn đề bằng cách dùng định lý Bayes:

- **Thay vì cố tính trực tiếp $P(C|X)$**, ta tính các thành phần **dễ ước lượng hơn**:
  
$$
P(C|X) \propto P(C) \cdot \prod_{i=1}^{n} P(x_i | C)
$$

Trong đó:

- $P(C)$: xác suất tiên nghiệm — ước lượng bằng tần suất văn bản thuộc lớp $C$ trong tập huấn luyện
- $P(x_i | C)$: xác suất từ $x_i$ xuất hiện trong lớp $C$ ( trong tập train)  

---

### Cách ước lượng $P(x_i \mid C)$

Giả sử:

- $N_{x_i, C}$: số lần từ $x_i$ xuất hiện trong tất cả các văn bản thuộc lớp $C$
- $N_C$: tổng số từ trong tất cả văn bản của lớp $C$
- $|V|$: số lượng từ trong từ điển (vocabulary)

Công thức ước lượng:

$$
P(x_i \mid C) = \frac{N_{x_i, C} + 1}{N_C + |V|}
$$

> Đây là công thức có **Laplace smoothing** (cộng 1), giúp tránh tình trạng xác suất bằng 0 nếu từ chưa từng xuất hiện trong lớp.

---

### Ước lượng $P(C)$

$$
P(C) = \frac{\text{số văn bản thuộc lớp } C}{\text{tổng số văn bản}}
$$

---


## Tại sao gọi là "Naive"?

Mô hình được gọi là **naive (ngây thơ)** vì giả định rằng các đặc trưng là **độc lập tuyệt đối với nhau**, điều này hiếm khi đúng trong thực tế. Tuy nhiên, trên thực tế, mô hình vẫn hoạt động rất hiệu quả trong nhiều tình huống — đặc biệt là với dữ liệu văn bản có nhiều đặc trưng (từ vựng) nhưng ít tương quan trực tiếp.

---

## Ví dụ minh họa

Giả sử bạn đang phân loại email thành **spam** hoặc **không spam**, và mỗi từ xuất hiện trong email được xem như một đặc trưng $x_i$. Mô hình Naive Bayes sẽ so sánh:

$$
P(\text{Spam}) \cdot \prod_i P(x_i | \text{Spam}) \quad \text{và} \quad P(\text{Ham}) \cdot \prod_i P(x_i | \text{Ham})
$$

Email sẽ được gán vào lớp nào có xác suất lớn hơn.
![Image Spam/Ham](/images/M02W02/spam.png)

---

## 1. Gaussian Naive Bayes và Định lý Giới hạn Trung tâm (CLT)

Trong nhiều bài toán học máy, đặc biệt khi xử lý dữ liệu **liên tục** (như chiều cao, cân nặng, nhiệt độ...), mô hình **Gaussian Naive Bayes** là lựa chọn phù hợp.

---

### Liên hệ với Định lý Giới hạn Trung tâm (Central Limit Theorem)

**CLT** phát biểu rằng:  
> Trung bình của một số lượng lớn các biến ngẫu nhiên độc lập và có cùng phân phối sẽ xấp xỉ phân phối chuẩn (Gaussian), dù các biến ban đầu không phải là chuẩn.
![Image CLT](/images/M02W02/clt.png)
⟶ Điều này giải thích vì sao nhiều hiện tượng tự nhiên tuân theo phân phối chuẩn. Ví dụ: điểm thi, chiều cao người, nhiễu tín hiệu,...

Do đó, khi xử lý dữ liệu thực, ta **có thể giả định** rằng mỗi đặc trưng liên tục $x_i$ **tuân theo phân phối chuẩn** khi biết nhãn lớp $C$.
Ví dụ :
![Example CLT](/images/M02W02/example.png)
---

### Gaussian Naive Bayes là gì?

Đây là biến thể của Naive Bayes trong đó ta **giả định các đặc trưng liên tục** có phân phối Gaussian (chuẩn) trong từng lớp $C$.

Cụ thể:

$$
P(x_i | C) = \frac{1}{\sqrt{2\pi \sigma_C^2}} \cdot \exp\left( -\frac{(x_i - \mu_C)^2}{2\sigma_C^2} \right)
$$

Trong đó:

- $\mu_C$: trung bình của đặc trưng $x_i$ trong lớp $C$
- $\sigma_C^2$: phương sai của đặc trưng \(x_i\) trong lớp $C$

→ Các tham số $\mu_C$ và $\sigma_C^2$ được **tính từ tập huấn luyện**, giống như trong thống kê.

---

### Cách hoạt động:

1. Với mỗi đặc trưng $x_i$ và mỗi lớp $C$:
   - Tính $\mu_C$, $\sigma_C$ từ tập huấn luyện
2. Khi phân loại mẫu mới $X = (x_1, x_2, ..., x_n)$:
   - Tính $P(x_i | C)$ bằng công thức Gaussian
   - Tính xác suất hậu nghiệm:
     
     $$
     P(C|X) \propto P(C) \cdot \prod_{i} P(x_i | C)
     $$

3. Chọn lớp $C$ có xác suất cao nhất

---

## Khi nào nên dùng Gaussian hay Multinomial Naive Bayes?

Việc lựa chọn biến thể nào của Naive Bayes phụ thuộc rất nhiều vào **kiểu dữ liệu** và **mức độ phong phú của tập huấn luyện**.

### Multinomial Naive Bayes — "Đếm trực tiếp"

Phù hợp khi:

- Dữ liệu là **rời rạc**, chẳng hạn như: từ ngữ trong văn bản, số lần xuất hiện từ
- Tập huấn luyện **đủ lớn**, từ vựng không quá hiếm
- Bạn có thể **đếm trực tiếp tần suất** của các từ với độ tin cậy

→ Đây là lựa chọn hàng đầu trong **NLP** khi dữ liệu phong phú (ví dụ: email, tin nhắn, bài báo...)

### Gaussian Naive Bayes — "Tổng quát hóa từ ít dữ liệu"

Phù hợp khi:

- Đặc trưng là **liên tục** (số thực): ví dụ như chiều cao, điểm thi, nhiệt độ...
- Dữ liệu **không lặp lại nhiều**, khó đếm chính xác
- Tập huấn luyện **ít dữ liệu**, hoặc đặc trưng **không rời rạc**

→ Trong trường hợp này, giả định đặc trưng theo **phân phối chuẩn (normal distribution)** giúp mô hình khái quát hóa tốt hơn, chỉ cần ước lượng $\mu$, $\sigma^2$ cho mỗi đặc trưng trong từng lớp.

---

### So sánh nhanh

| Tình huống | Biến thể phù hợp | Lý do |
|-----------|------------------|-------|
| Văn bản, từ ngữ, đếm số lần xuất hiện | Multinomial NB | Dữ liệu rời rạc, dễ đếm, có thể smoothing |
| Dữ liệu số thực: giá trị, thời gian, cảm biến | Gaussian NB | Ước lượng bằng phân phối chuẩn |
| Dữ liệu ít, không đủ đếm đầy đủ | Gaussian NB | Khái quát tốt với $\mu, \sigma$ |
| Từ vựng lớn, dữ liệu ít |  Multinomial dễ bị sparse | Không đủ đếm hết từ → xác suất thấp |

---

### Tổng kết

> **Multinomial Naive Bayes**: tốt khi dữ liệu rời rạc, có thể đếm được  
> **Gaussian Naive Bayes**: tốt khi dữ liệu liên tục, hoặc ít dữ liệu

---
