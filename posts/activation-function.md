---
title: "Activation Function trong Neural Networks"
date: "2025-10-06"
---


## 1. Vì sao cần hàm kích hoạt (Activation Function)?
![Minh họa mạng neural networks](/images/linearCombination.png)

Trong mạng neural, nếu không sử dụng hàm kích hoạt phi tuyến tính giữa các lớp, mỗi lớp chỉ thực hiện phép biến đổi tuyến tính, Tổ hợp của các phép biến đổi tuyến tính cũng chỉ là 1 phép biến đổi tuyến tính. Khi đó, dù có nhiều tầng, toàn bộ mạng chỉ tương đương với một hàm tuyến tính duy nhất:

$$
f(x) = W_3(W_2(W_1 x + b_1) + b_2) + b_3 = W x + b
$$

Điều này có nghĩa là mạng không thể học được các mối quan hệ **phi tuyến tính phức tạp** trong dữ liệu — điều thiết yếu cho các bài toán thực tế như phân loại ảnh, nhận diện giọng nói, xử lý ngôn ngữ tự nhiên, v.v.

---

## 2. Ví dụ kinh điển: Bài toán XOR
![Minh họa bài toán XOR không tuyến tính phân tách](/images/xor.png)

Hàm XOR là một ví dụ đơn giản nhưng nổi tiếng để minh họa sự cần thiết của phi tuyến tính trong mô hình học máy.

| x₁ | x₂ | XOR(x₁, x₂) |
|----|----|-------------|
| 0  | 0  | 0           |
| 0  | 1  | 1           |
| 1  | 0  | 1           |
| 1  | 1  | 0           |

Tập dữ liệu trên **không thể phân tách bằng một đường thẳng** — tức là không tuyến tính phân tách được. Do đó, một mạng không có activation function sẽ **không thể học được hàm XOR**.

---

## 3. Giải pháp: Thêm hàm kích hoạt phi tuyến

Để giải quyết bài toán XOR và các bài toán phi tuyến khác, ta cần chèn hàm kích hoạt (activation function) vào giữa các tầng:

$$
\begin{aligned}
h &= \text{ReLU}(W_1 x + b_1) \\
y &= W_2 h + b_2
\end{aligned}
$$

Tầng ẩn (`h`) được phi tuyến hóa nhờ ReLU (hoặc các hàm khác), giúp mô hình biểu diễn tốt hơn các quan hệ phức tạp.

---

- **ReLU (Rectified Linear Unit)**:  
  $$
  \text{ReLU}(x) = \max(0, x)
  $$
  Đơn giản, tính toán nhanh, tránh gradient vanish.  
  Tuy nhiên, nếu đầu vào nhỏ hơn 0, gradient sẽ bằng 0 → dễ gây hiện tượng "chết neuron".

  ```python
  import numpy as np
  def relu(x):
      return np.maximum(0, x)
  ```

---

- **Leaky ReLU**:  
  $$
  \text{LeakyReLU}(x) =
  \begin{cases}
  x, & \text{nếu } x > 0 \\
  \alpha x, & \text{nếu } x \leq 0
  \end{cases}
  $$
  Với $\alpha$ là một hằng số nhỏ (thường là 0.01).  
  Cho phép một lượng nhỏ gradient khi $x < 0$, giúp neuron không bị “chết”.

  ```python
  def leaky_relu(x, alpha=0.01):
      return np.where(x > 0, x, alpha * x)
  ```

---

- **ELU (Exponential Linear Unit)**:  
  $$
  \text{ELU}(x) =
  \begin{cases}
  x, & \text{nếu } x > 0 \\
  \alpha (e^x - 1), & \text{nếu } x \leq 0
  \end{cases}
  $$
  Làm mượt đầu ra khi $x < 0$ và giúp trung bình đầu ra gần 0 → mạng hội tụ nhanh hơn.  
  Nhưng tốn tính toán hơn ReLU.

  ```python
  def elu(x, alpha=1.0):
      return np.where(x > 0, x, alpha * (np.exp(x) - 1))
  ```

---

- **Sigmoid**:  
  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$
  Dễ hiểu, đầu ra trong khoảng (0, 1), phù hợp cho bài toán phân loại nhị phân.  
  Nhược điểm là gây **vanishing gradient** khi $x$ quá lớn hoặc quá nhỏ.

  ```python
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))
  ```

---

- **Tanh**:  
  $$
  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  $$
  Giống sigmoid nhưng có đầu ra trong khoảng (-1, 1) và trung bình tại 0 → tốt hơn sigmoid.  
  Vẫn có thể bị vanishing gradient khi đầu vào lớn về độ lớn.

  ```python
  def tanh(x):
      return np.tanh(x)
  ```

---

- **Swish**:  
  $$
  \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
  $$
  Được đề xuất bởi Google.  
  Trơn (smooth), không bị chết như ReLU và cho hiệu suất cao hơn trong nhiều mạng sâu.  
  Tuy nhiên, tính toán phức tạp hơn một chút.

  ```python
  def swish(x):
      return x * sigmoid(x)
  ```

---


## 5. Kết luận

Hàm kích hoạt là một **thành phần không thể thiếu** trong neural networks. Chúng cho phép mô hình:

- Biểu diễn các quan hệ phi tuyến tính.
- Giải quyết các bài toán mà mô hình tuyến tính không làm được.
- Tăng khả năng học biểu diễn phức tạp và hiệu quả hơn.

Việc lựa chọn đúng hàm kích hoạt cũng ảnh hưởng đến hiệu suất và độ hội tụ của mạng trong quá trình huấn luyện.

---
