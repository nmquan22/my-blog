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
  ![ReLU image](/images/ReLU.png)
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
  ![ELU image](/images/ELU.png)
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
  ![sigmoid image](/images/sigmoid.png)
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

## 5. Khai triển taylor 

Khai triển Taylor (Taylor Expansion) là một kỹ thuật toán học dùng để xấp xỉ một hàm số bằng đa thức quanh một điểm. Trong học máy và deep learning, khai triển Taylor thường được dùng để:

- Giải thích cách một hàm số thay đổi xung quanh điểm gốc.
- Phân tích gradient và tối ưu hàm mất mát.

**Công thức tổng quát:**

$$
f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f^{(3)}(a)}{3!}(x - a)^3 + \dots
$$

Trong đó:
- $f^{(n)}(a)$ là đạo hàm bậc $n$ tại điểm $a$.
- Với $a = 0$, ta được khai triển **Maclaurin**.

**Ví dụ**:

- $e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots$
- $\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots$

**Code cho hàm sin**:
```python
import math

def factorial_fcn(n):
    """Tính giai thừa của n"""
    return math.factorial(n)

def approx_sin(x, n):
    """
    Approximate the sine of x using the Taylor series expansion.

    Parameters:
    x (float): The input angle in radians.
    n (int): Number of terms in the Taylor series expansion.

    Returns:
    float: Approximate value of sin(x) using n+1 terms.
    """
    sin_approx = 0
    for i in range(n + 1):
        coef = (-1) ** i
        num = x ** (2 * i + 1)
        denom = factorial_fcn(2 * i + 1)
        sin_approx += coef * (num / denom)

    return sin_approx

# Ví dụ sử dụng
print(round(approx_sin(x=3.14, n=10), 4))

```
---

## 6. F-Score matrix & F1-Score

F-Score là chỉ số cân bằng giữa **Precision** và **Recall**, thường dùng trong bài toán phân loại, đặc biệt là với dữ liệu mất cân bằng.

### **Confusion Matrix (Ma trận nhầm lẫn)**

|               | **Dự đoán: Positive** | **Dự đoán: Negative** |
|---------------|------------------------|------------------------|
| **Thực tế: Positive** | TP (True Positive)      | FN (False Negative)     |
| **Thực tế: Negative** | FP (False Positive)     | TN (True Negative)      |

---

### **Precision (Độ chính xác)**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

---

### **Recall (Độ bao phủ)**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

---

### **F1-Score**

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---


## 7. Kết luận

Hàm kích hoạt là một **thành phần không thể thiếu** trong neural networks. Chúng cho phép mô hình:

- Biểu diễn các quan hệ phi tuyến tính.
- Giải quyết các bài toán mà mô hình tuyến tính không làm được.
- Tăng khả năng học biểu diễn phức tạp và hiệu quả hơn.

Việc lựa chọn đúng hàm kích hoạt cũng ảnh hưởng đến hiệu suất và độ hội tụ của mạng trong quá trình huấn luyện.

---
