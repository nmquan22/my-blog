---
title: 'Python Word Suggestion and Algorithms'
date: '2025-06-15'
---

## Sliding Window Maximum (Getting Max Over Kernel)

Cho danh sách số nguyên `num_list` và một cửa sổ trượt (sliding window) có độ dài `k`, bài toán yêu cầu tìm giá trị lớn nhất trong mỗi lần trượt cửa sổ qua danh sách.

---

### Naive Approach (Brute Force) - $O(n \cdot k)$

Phương pháp ngây thơ là:

- Duyệt từng vị trí bắt đầu của cửa sổ.
- Dùng hàm `max()` để lấy giá trị lớn nhất trong mỗi cửa sổ con có kích thước `k`.

```python
def max_kernel_naive(num_list, k):
    result = []
    for i in range(len(num_list) - k + 1):
        window = num_list[i:i+k]
        result.append(max(window))
    return result

# Ví dụ
num_list = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(max_kernel_naive(num_list, k))  # Output: [3, 3, 5, 5, 6, 7]
```

Độ phức tạp thời gian:

$$
T(n) = O(n \cdot k)
$$
---

### Optimized Approach Using Deque - $O(n)$
Ý tưởng ban đầu ta định xây dựng 1 dãy `max` của các sliding window , trong đó  max[i+1] = max(max[i],nums[i+k])
nhưng lỡ nums[i] là `max` thì sao , lỡ trong đoạn [i,i+k-1] có thêm 1 số cũng bằng nums[i] thì thuật toán này không giải quyết được.

Vậy nên chúng ta nghĩ ra ý tưởng mới không chỉ giữ lại max mà giữ lại những điểm có khả năng làm max trong tương lai nếu bé hơn nums[i] chỉ số hiện tại thì ta có thể bỏ nó đi.

Ta sử dụng `deque` để duy trì chỉ số của các phần tử có thể là lớn nhất trong từng cửa sổ. Mỗi phần tử được thêm và loại bỏ khỏi deque nhiều nhất một lần.Do vậy độ phức tạp là O(n).

```python
from collections import deque

def max_kernel_optimized(nums, k):
    q = deque()
    result = []

    for i in range(len(nums)):
        # Loại bỏ phần tử đã ra khỏi cửa sổ
        # q để lưu lại vị trí nếu quá xa thì bỏ nó đi cụ thể bé hơn i-k+1 
        while q and q[0] <= i - k:
            q.popleft()

        # Loại bỏ phần tử nhỏ hơn phần tử hiện tại
        # do vậy những phần tử phía trước của q đều lớn hơn nums[i] nếu có do vậy nums[q[i]] giảm dần 
        while q and nums[q[-1]] < nums[i]:
            q.pop()

        q.append(i)

        # Ghi lại kết quả khi đủ k phần tử
        if i >= k - 1:
            result.append(nums[q[0]])

    return result

# Ví dụ
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(max_kernel_optimized(nums, k))  # Output: [3, 3, 5, 5, 6, 7]
```

Độ phức tạp thời gian:

$$
T(n) = O(n)
$$

---

### So sánh hiệu năng

| Phương pháp           | Độ phức tạp       | Ưu điểm                  | Nhược điểm                    |
|-----------------------|-------------------|---------------------------|-------------------------------|
| Naive (Brute Force)   | $O(n \cdot k)$     | Dễ hiểu, dễ cài đặt       | Chạy chậm với dữ liệu lớn     |
| Deque Tối Ưu          | $O(n)$             | Hiệu suất cao, tối ưu     | Cần hiểu logic deque phức tạp |

---

## Character Counting

### Description

**Problem**:  
Viết một thuật toán để đếm số lần xuất hiện của từng chữ cái trong một từ. Kết quả trả về dưới dạng một dictionary với:
- **Key** là chữ cái.
- **Value** là số lần chữ cái đó xuất hiện.

**Input**:  
- Một chuỗi ký tự (gồm các chữ cái từ `a-z` hoặc `A-Z`)

**Output**:  
- Một dictionary biểu diễn số lần xuất hiện của từng chữ cái trong từ đó.

### Naive Approach – Using Dictionary

Chúng ta sẽ sử dụng một dictionary đơn giản để duyệt qua từng ký tự và tăng giá trị đếm tương ứng.

```python
def count_characters(word):
    count = {}
    for char in word:
        char = char.lower()  # chuyển về cùng kiểu chữ
        if char.isalpha():   # chỉ đếm chữ cái
            if char in count:
                count[char] += 1
            else:
                count[char] = 1
    return count

# Ví dụ
print(count_characters("OpenAI"))  
```
---

## Word Counting 

### Description

 **Word Counting** là bài toán yêu cầu bạn đếm số lần xuất hiện của mỗi từ trong một file văn bản `.txt`.

- **Input**: Đường dẫn đến file `.txt`
- **Output**: Một `dictionary` với `{word: count}`
- **Giả sử**: Các từ chỉ bao gồm chữ cái [a-zA-Z]

---

###  Text Preprocessing

Trước khi đếm, ta cần chuẩn hóa văn bản:

- Chuyển toàn bộ sang chữ thường (lowercasing)
- Loại bỏ dấu câu (punctuation removal)
- Tách câu thành danh sách các từ (tokenization)

```python
import string

def preprocess_text(sentence):
    """
    Tiền xử lý:
    - Lowercase toàn bộ
    - Loại bỏ dấu câu
    - Tách thành danh sách từ
    """
    sentence = sentence.lower()
    for p in string.punctuation:
        sentence = sentence.replace(p, '')
    words = sentence.split()
    return words

# Kiểm tra
s = "I love AI. AI is not easy,"
print(preprocess_text(s))  
# ['i', 'love', 'ai', 'ai', 'is', 'not', 'easy']
```

---

## Word Counting: Manual Implementation

Đếm từng từ bằng cách lặp qua danh sách từ đã được xử lý.

```python
def count_word(data_path):
    with open(data_path, 'r') as f:
        document = f.read()
        words = preprocess_text(document)

    counter = {}
    for word in words:
        if word in counter:
            counter[word] += 1
        else:
            counter[word] = 1
    return counter

# Sử dụng
data_path = './P1_data.txt'
result = count_word(data_path)
print(result.get('man', 0))
```

---
## Levenshtein Distance

###  Motivation

Trong xử lý ngôn ngữ tự nhiên, một trong những vấn đề phổ biến là xử lý sai chính tả hoặc tìm từ gần đúng. Ví dụ:

- Người dùng nhập `"presenteton"` thay vì `"presentation"`.
![Levenshtein image](/images/LevenshteinMotivation.png)

**Câu hỏi đặt ra:**  
> Làm sao đo lường được "độ giống nhau" hay "khoảng cách" giữa 2 chuỗi?

Đó là lúc **Levenshtein Distance** ra đời.

---

### Định nghĩa

**Levenshtein Distance** là số phép biến đổi tối thiểu (thêm, xóa, thay ký tự) để biến đổi chuỗi `s1` thành `s2`.

Ví dụ:  
`presenteton` → `presentation`: cần 2 thao tác (thêm `"a"`, xóa `"e"`)

---

## Thuật toán Quy hoạch Động

Gọi `dp[i][j]` là số phép biến đổi tối thiểu để biến `i` phần tử đầu tiên của xâu thứ nhất thành `j` phần tử đầu tiên của xâu thứ hai. Ta xây dựng công thức quy hoạch động như sau:

Xét phép biến đổi `i+1` phần tử đầu tiên của xâu thứ nhất thành `j+1` phần tử đầu tiên của xâu thứ hai. Giả sử hai xâu là `a` và `b`:

- Nếu phần tử `a[i+1]` và `b[j+1]` **bằng nhau**, khi đó phép biến đổi ít nhất là phép biến đổi `i` phần tử của `a` thành `j` phần tử của `b`.

- Nếu `a[i+1]` **khác** `b[j+1]`, ta xét tất cả khả năng có thể có của `a[i+1]`:

  - **Xóa**:  
    Khi đó phép biến đổi trở thành `i` phần tử đầu tiên của `a` thành `j+1` phần tử đầu tiên của `b`  
    ⇒ `dp[i][j+1] + 1`

  - **Thay đổi** `a[i+1]` thành phần tử `b` nào đó:  
    - Nếu là `b[j+1]` ⇒ `dp[i][j] + 1`  
    - Nếu là `b[k]` với `k ≤ j`:  
      (Chứng minh logic cho phần chèn `b[j+1]` vào `a` thu được `dp[i+1][j]`)  
      Từ `b[k+1]` tới `b[j+1]` sẽ bị xóa đi (do `b` không thay đổi nên điều này tương đương thêm bấy nhiêu phần tử vào đuôi của `a`)  
      ⇒ Phép biến đổi trở thành `i` phần tử của `a` thành `k-1` phần tử của `b`  
      ⇒ `dp[i][k-1] + (j - k + 2)`

Ta xét `dp[i+1][j]` là phép biến đổi `i+1` phần tử của `a` thành `j` phần tử của `b`. Xét một cách biến đổi như sau:

- Biến `i` phần tử đầu tiên của `a` thành `k-1` phần tử của `b`
- Biến `a[i+1] → b[k]`, xóa `b[k+1]` tới `b[j]` (giống như thêm vào đuôi của `a`)

Vì đây cũng là một phép biến đổi `i+1` phần tử của `a` thành `j` phần tử của `b` nên:

$$
dp[i][k-1] + (j - k + 1) \le dp[i+1][j]
$$

Hơn nữa, trong phép biến đổi `i+1` phần tử của `a` thành `j` phần tử của `b`, nếu `a[i+1]` không là `b[k]` với `k ≤ j` tức `a[i+1]` sẽ bị xóa khỏi phép biến đổi này  
khi đó phép biến đổi trở thành biến đổi `i` phần tử của `a` thành `j` phần tử của `b` => `dp[i][j] + 1`  
cho nên 
$$
 dp[i+1][j]+1  = min(dp[i][j] + 1 ,dp[i][k-1] + (j - k + 1)) + 1 
$$
Hơn nữa 
$$
dp[i][j] + 1 \le dp[i][j] + 2 
$$
có trong trường hợp thay đổi `a[i+1]` thành `b[j+1]`
Vì vậy, ta có công thức:

$$
dp[i][j] = 
\begin{cases} 
dp[i-1][j-1] & \text{nếu } a[i] = b[j] \\\\
1 + \min(dp[i-1][j],\ dp[i][j-1],\ dp[i-1][j-1]) & \text{nếu } a[i] \ne b[j]
\end{cases}
$$

```python
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    # Khởi tạo: biến đổi chuỗi rỗng
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    # Tính khoảng cách từng cặp con chuỗi
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # xóa
                dp[i][j-1] + 1,      # thêm
                dp[i-1][j-1] + cost  # thay
            )
    return dp[m][n]

