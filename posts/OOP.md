---
title: "Object-Oriented Programming"
date: "2025-06-27"
---

## 1. Lập trình Hướng Đối Tượng (OOP)

**OOP (Object-Oriented Programming)** là phương pháp lập trình mô phỏng các thực thể trong thế giới thực thông qua:
- **Class (lớp)** và **Object (đối tượng)**
- 4 tính chất cốt lõi: *Encapsulation*, *Abstraction*, *Inheritance*, *Polymorphism*
![Dog object](/images/W03/dog.png)
---

### Class và Object – Mô hình hóa đối tượng thực

```python
class Dog:
    def __init__(self, name, size, age, color):
        self.name = name
        self.size = size
        self.age = age
        self.color = color

    def eat(self):
        return 'Chicken' if self.age <= 1 else 'Fish'
```

---

### Encapsulation (Đóng gói)

Ẩn chi tiết nội bộ của đối tượng, chỉ cho phép truy cập qua phương thức.

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # private

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def get_balance(self):
        return self.__balance
```

---

### Abstraction (Trừu tượng) với ABC
Lớp trừu tượng (abstract) định nghĩa giao diện chung, ẩn đi chi tiết triển khai cụ thể.

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def start_engine(self):
        pass

class Car(Vehicle):
    def start_engine(self):
        return "Car engine started"

class Motorbike(Vehicle):
    def start_engine(self):
        return "Motorbike engine started"
```

---

### Inheritance (Kế thừa)

Lớp con kế thừa thuộc tính và phương thức từ lớp cha.

```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def compute_salary(self):
        return self.salary

class Manager(Employee):
    def __init__(self, name, salary, bonus):
        super().__init__(name, salary)
        self.bonus = bonus

    def compute_salary(self):
        return self.salary + self.bonus
```

---

### Polymorphism (Đa hình) với ABC

Cho phép gọi phương thức giống nhau (`area`) trên nhiều lớp khác nhau, hành vi được xác định tại runtime.  
Ví dụ khi tính tổng diện tích của các hình khác nhau : tam giác , hình vuông, hình tròn nhập từ input mà không cần sử dụng if -else dài dòng  

```python
from abc import ABC, abstractmethod
import math

# Lớp trừu tượng
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height
```

#### Áp dụng đa hình – không cần `if/else`:

```python
shapes = [
    Square(4),
    Circle(3),
    Triangle(6, 2)
]

for shape in shapes:
    print(f"{type(shape).__name__} area: {shape.area():.2f}")
```

---

### Tổng kết 4 tính chất OOP

| Tính chất        | Ý nghĩa                                                        | Lợi ích                                                  |
|------------------|----------------------------------------------------------------|-----------------------------------------------------------|
| Encapsulation    | Ẩn dữ liệu và chỉ cho phép truy cập qua phương thức           | Bảo vệ dữ liệu, dễ bảo trì                               |
| Abstraction      | Ẩn chi tiết, chỉ lộ ra những gì cần thiết                      | Đơn giản hóa, rõ ràng                                    |
| Inheritance      | Lớp con kế thừa và mở rộng lớp cha                            | Tái sử dụng, giảm lặp mã                                  |
| Polymorphism     | Cùng một phương thức, hoạt động khác nhau trên mỗi class      | Mở rộng linh hoạt, không sửa code cũ                     |


---

## 2. OOP trong PyTorch

### nn.Module & `forward()`

- `nn.Module` là lớp cha trừu tượng đại diện cho mọi mô hình học sâu trong PyTorch.
- Khi định nghĩa mô hình mới, ta kế thừa từ `nn.Module` và ghi đè phương thức `forward()` để xác định cách dữ liệu đi qua mô hình.
Ví dụ như hàm sigmoid:  
```python
import torch
import torch.nn as nn

# Định nghĩa lớp Sigmoid kế thừa từ nn.Module
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()  # gọi constructor lớp cha

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))  # sigmoid thủ công
```
### Vì sao nên dùng `nn.Module` thay vì hàm ngoài cho sigmoid?

#### So sánh nhanh

| Hàm ngoài `sigmoid(x)` | Lớp `Sigmoid(nn.Module)` |
|-------------------------|----------------------------|
| Không kế thừa gì        | Kế thừa toàn bộ `nn.Module` |
| Không dùng được `.to()`, `.eval()` | Dùng đầy đủ hệ sinh thái PyTorch |
| Không dễ tái sử dụng    | Có thể dùng trong `Sequential`, `torch.fx`, kế thừa dễ |
| Không mở rộng được      | Dễ thêm tính năng qua kế thừa |
| Không hỗ trợ lưu/truy cập | Hỗ trợ `state_dict()` & `.load_state_dict()` |

---

#### Lợi ích khi dùng `nn.Module`

- **Tương thích PyTorch**: Di chuyển sang GPU, lưu mô hình, eval/train mode, v.v.
- **Tổ chức rõ ràng (Encapsulation)**: Dễ quản lý logic sigmoid, gắn thêm các tính năng khác.
- **Mở rộng linh hoạt (Inheritance)**: Có thể viết lớp `ClippedSigmoid`, `LearnableSigmoid` kế thừa dễ dàng.
- **Dễ dùng trong mô hình lớn**: Dùng được với `nn.Sequential`, `nn.ModuleList`, `torch.jit`, `torch.fx`.
- **Tuân thủ OOP**: Giúp mô hình hóa rõ ràng, dễ kiểm thử, dễ bảo trì hơn.

---

##  3. OOP – Quản lý người dùng (User Management)

### Cấu trúc class:

```text
         Person
  ┌────────┴────────┐
Student   Doctor   Teacher
```

- `Person` có: `name`, `yob`, `describe()`
- `Student`: thêm `grade`
- `Doctor`: thêm `nspecialist`
- `Teacher`: thêm `subject`

### Code mẫu:

```python
class Person:
    def __init__(self, name, yob):
        self.name = name
        self.yob = yob

    def describe(self):
        return f"{self.name}, born {self.yob}"

class Student(Person):
    def __init__(self, name, yob, grade):
        super().__init__(name, yob)
        self.grade = grade

class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name, yob)
        self.specialist = specialist

class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name, yob)
        self.subject = subject
```

---

## 5. Stack (Ngăn xếp)

Cấu trúc dữ liệu LIFO (Last In First Out) – phần tử thêm sau cùng sẽ được lấy ra đầu tiên.

Thao tác chính:

push(x): thêm phần tử vào đỉnh stack.

pop(): loại bỏ phần tử ở đỉnh stack.

top() hoặc peek(): xem giá trị đỉnh stack mà không loại bỏ.
Lưu ý: Khi sử dụng Stack, cần kiểm tra:

Tràn ngăn xếp (Overflow): xảy ra nếu gọi push() khi stack đã đầy (vượt quá capacity).
Rỗng ngăn xếp (Underflow): xảy ra nếu gọi pop() hoặc top() khi stack đang rỗng.
![Image Stack](/images/W03/stack.png)

### Ứng dụng:
Phân tích biểu thức có dấu ngoặc: kiểm tra dấu ngoặc đóng-mở có khớp không (ví dụ ((a+b)*c)).

Trình quản lý cửa sổ hoặc tab: khi bấm nút Back/Close, sẽ đóng cái mở gần nhất (giống cơ chế Ctrl+Z – hoàn tác).

Đệ quy: ngầm sử dụng stack để lưu trạng thái gọi hàm.

--- 

### Time Complexity:
|Thao tác |	Độ phức tạp|
|---------|-------------|
|push(x)|	O(1)|
|pop()|	O(1)|
|top()|	O(1)|
---
### Cài đặt:
```python 
class MyStack:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__stack = []

    def is_empty(self):
        return len(self.__stack) == 0

    def is_full(self):
        return len(self.__stack) == self.__capacity

    def push(self, value):
        if self.is_full():
            raise Exception("Overflow")  # Stack đầy
        self.__stack.append(value)

    def pop(self):
        if self.is_empty():
            raise Exception("Underflow")  # Stack rỗng
        return self.__stack.pop()

    def top(self):
        if self.is_empty():
            print("Stack is empty")
            return None
        return self.__stack[-1]
```

---

## 6. Queue (Hàng đợi)

Cấu trúc FIFO (First In First Out) – phần tử thêm vào đầu tiên sẽ được lấy ra đầu tiên.

Gồm hai con trỏ: front và rear

Thao tác chính:

-enqueue(x): thêm phần tử vào cuối hàng.

-dequeue(): loại bỏ phần tử ở đầu hàng.
Lưu ý khi dùng Queue:

Overflow (Tràn hàng đợi): xảy ra khi gọi enqueue() trong khi queue đã đầy (đạt tới capacity).

Underflow (Hàng đợi rỗng): xảy ra khi gọi dequeue() hoặc front() trong khi queue không có phần tử nào.
![Image Queue](/images/W03/queue.png)

### Ứng dụng:
Xử lý theo thứ tự: như xếp hàng giao dịch ngân hàng, thanh toán, hoặc hệ thống in ấn (in tài liệu theo thứ tự yêu cầu).

Hệ thống điều phối tác vụ (task scheduling): CPU, hàng đợi in, message queue.

---

### Time Complexity:
|Thao tác |	Độ phức tạp|
|---------|-------------|
|enqueue(x)|	O(1)|
|dequeue()| O(1)|
|front()/peek()|	O(1)|

### Cài đặt:
```python 
class MyQueue:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__queue = []

    def is_empty(self):
        return len(self.__queue) == 0

    def is_full(self):
        return len(self.__queue) == self.__capacity

    def enqueue(self, value):
        if self.is_full():
            raise Exception("Overflow")  # Hàng đợi đầy
        self.__queue.append(value)

    def dequeue(self):
        if self.is_empty():
            raise Exception("Underflow")  # Hàng đợi rỗng
        return self.__queue.pop(0)  # Xóa phần tử đầu tiên (FIFO)

    def front(self):
        if self.is_empty():
            print("Queue is empty")
            return None
        return self.__queue[0]
```


---

