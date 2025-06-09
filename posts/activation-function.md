---
title: "Activation Function trong Neural Networks"
date: "2025-10-06"
---


## 1. Vì sao cần hàm kích hoạt (Activation Function)?

Trong mạng neural, nếu không sử dụng hàm kích hoạt phi tuyến tính giữa các lớp, mỗi lớp chỉ thực hiện phép biến đổi tuyến tính. Khi đó, dù có nhiều tầng, toàn bộ mạng chỉ tương đương với một hàm tuyến tính duy nhất:

$$
f(x) = W_3(W_2(W_1 x + b_1) + b_2) + b_3 = W x + b
$$

Điều này có nghĩa là mạng không thể học được các mối quan hệ **phi tuyến tính phức tạp** trong dữ liệu — điều thiết yếu cho các bài toán thực tế như phân loại ảnh, nhận diện giọng nói, xử lý ngôn ngữ tự nhiên, v.v.

---

## 2. Ví dụ kinh điển: Bài toán XOR

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

## 4. Các hàm kích hoạt phổ biến

- **ReLU**:  
  $$ 
  \max(0, x) 
  $$  
  Đơn giản, tính toán nhanh, tránh gradient vanish

- **Sigmoid**:  
  $$ 
  \frac{1}{1 + e^{-x}} 
  $$  
  Dễ hiểu nhưng có thể gây vanishing gradient

- **Tanh**:  
  $$ 
  \tanh(x) 
  $$  
  Giá trị từ -1 đến 1, trung bình tại 0

---

## 5. Kết luận

Hàm kích hoạt là một **thành phần không thể thiếu** trong neural networks. Chúng cho phép mô hình:

- Biểu diễn các quan hệ phi tuyến tính.
- Giải quyết các bài toán mà mô hình tuyến tính không làm được.
- Tăng khả năng học biểu diễn phức tạp và hiệu quả hơn.

Việc lựa chọn đúng hàm kích hoạt cũng ảnh hưởng đến hiệu suất và độ hội tụ của mạng trong quá trình huấn luyện.

---
