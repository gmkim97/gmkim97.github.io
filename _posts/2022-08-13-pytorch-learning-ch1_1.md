---
title: PyTorch Learning_CH1_1. Tensor Manipulation
author: GMKim
date: 2022-08-13 11:45:00 +0900
categories: [Machine Learning, PyTorch]
tags: [Machine Learning, PyTorch, Tensor]
---

## Introduction

이 정리는 다음의 강의 노트를 보며 공부한 것을 기반으로 하고 있습니다.
- [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788)

이번 Chapter에서는 Tensor가 무엇인지, 어떻게 할당시키는지, 그리고 이를 다루는 여러 함수들에 대해 알아봅니다.  
<br>

---
## CH1_1. Tensor Manipulation - Part 1

### 1. What's Tensor?

- Tensor 정의  
    - 우선 Wikipedia에서는 Tensor를 다음과 같이 정의하고 있습니다.  
    ```
    선형대수학에서 텐서(Tensor)는 선형 관계를 나타내는 다중선형대수학의 대상이다.  
    ```
    사실 바로 와닿지 않기에 다음과 같이 단순하게 생각해 보기로 하였습니다.  
    ```
    텐서는 다차원 배열(Multi-dimensional Arrays)을 나타내는 도구이다.
    ```
    ![ch1_1](/assets/img/CH1/ch1_1.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

<br>  

- 1D with Numpy
    - 다차원 배열을 나타내는 데에 있어 저희는 Python에서 numpy라는 Library를 주로 사용해 왔습니다. 복습 겸 한번 다시 보도록 하죠.  
    먼저 단순히 1차원 행렬로 표현해 봅시다.

```python
import numpy as np
```
```python
arr = np.array([0.,1.,2.,3.,4.,5.,6.])
print(arr)
```
    [Output] 
    [0. 1. 2. 3. 4. 5. 6.]

```python
# Rank & Shape of 1D array
print("Rank of array : ", arr.ndim) ## Rank = Dimension
print("Shape of array : ", arr.shape) ## (7,) means (1,7)
print("Number of components in array : ", arr.size)
```
    [Output]
    Rank of array :  1
    Shape of array :  (7,)
    Number of components in array :  7

```python
# Components of 1D array
## Note that at Python range (a,b) : a 이상 b 미만
for i in range(0,7):
    print(arr[i])
print("---------------------------")
print(arr[-1]) ## Component num. -1 means last num.
print("---------------------------")
print(arr[0:2]) 
print(arr[2:-1])
print(arr[:5]) ## Start from beginning
print(arr[2:]) ## Stop at end
```
    [Output]
    0.0
    1.0
    2.0
    3.0
    4.0
    5.0
    6.0
    ---------------------------
    6.0
    ---------------------------
    [0. 1.]
    [2. 3. 4. 5.]
    [0. 1. 2. 3. 4.]
    [2. 3. 4. 5. 6.]

<br>  

- 2D with Numpy
    - 다음으로 numpy를 통해 2차원 행렬을 나타내 봅시다.

    ```python
    import numpy as np
    ```
    ```python
    arr_2 = np.array([[0.,1.,2.],[3.,4.,5.]])
    print(arr_2)
    ```
        [Output]
        [[0. 1. 2.]
        [3. 4. 5.]]

    ```python
    # Rank & Shape of 2D array
    print("Rank of array : ", arr_2.ndim)
    print("Shape of array : ", arr_2.shape)
    ```
        [Output]
        Rank of array :  2
        Shape of array :  (2, 3)
    
<br>  

- Array vs Tensor
    - 그렇다면 굳이 numpy의 array를 놔두고 torch를 이용해 tensor를 쓰는 이유는 무엇일까요?  
    이에 대해 찾아보던 중 Konstantinos Giorgas 님이 포스팅한 `What is the Difference Between NumPy Arrays and Tensorflow Tensors?` 글을 발견하게 되었습니다. (물론 저희는 Tensorflow가 아니라, Pytorch의 Tensor이긴 하지만, 어쨋든.)   
    - 즉, 요약하자면 Tensor는  
        1. GPU Accleration이 가능하다.  
        2. 자동으로 differentiation이 가능하다.
        3. 숫자 뿐만 아니라, string과 같은 type의 data 또한 다룰 수 있다.
    - 따라서, 단순히 Machine Learning 보다 부하가 더 걸리는 Deep Learning에 있어 Tensor를 사용하는 것이 더 적합하다고 볼 수 있겠군요.

<br>  

---
### 2. Tensor Allocation

- 1D with Pytorch
    - 이번엔 위에서 정의한 1차원 array들을 Pytorch의 torch library를 이용하여 재현해 보도록 하겠습니다.  
    한 가지 다른 점이라면, Output 되는 data가 단순히 array가 아닌 tensor라는 점이 되겠군요.

    ```python
    import torch
    ```
    ```python
    t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
    print(t)
    ```
        [Output]
        tensor([0., 1., 2., 3., 4., 5., 6.])
    
    ```python
    print("Rank of tensor : ", t.dim())
    print("Shape of tensor : ", t.shape)
    print("Shape of tensor : ", t.size()) ## This result is same as the result of t.shape
    ```
        [Output]
        Rank of tensor :  1
        Shape of tensor :  torch.Size([7])
        Shape of tensor :  torch.Size([7])

<br>  

- 2D with Pytorch
    - 이번엔 차원이 2인 array 입니다.

    ```python
    t_2 = torch.FloatTensor([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0],[10.0,11.0,12.0]])
    print(t_2)
    ```
        [Output]
        tensor([[ 1.,  2.,  3.],
                [ 4.,  5.,  6.],
                [ 7.,  8.,  9.],
                [10., 11., 12.]])



    ```python
    print("Rank of tensor : ", t_2.dim())
    print("Shape of tensor : ", t_2.shape)
    print("Shape of tensor : ", t_2.size())
    ```
        [Output]
        Rank of tensor :  2
        Shape of tensor :  torch.Size([4, 3])
        Shape of tensor :  torch.Size([4, 3])



    ```python
    # Slicing
    print(t_2[:,:])
    print(t_2[:2, :2])
    print(t_2[1:3, :])
    ## t_2[Row_index, Column_index]
    ```
        [Output]
        tensor([[ 1.,  2.,  3.],
                [ 4.,  5.,  6.],
                [ 7.,  8.,  9.],
                [10., 11., 12.]])
        tensor([[1., 2.],
                [4., 5.]])
        tensor([[4., 5., 6.],
                [7., 8., 9.]])

<br>  

- Size of Tensors
    - numpy의 array이든, torch의 tensor이든, 앞으로 함수를 사용하는데 있어 매우 중요한 것은 **Tensor의 크기**를 파악하는 것 입니다. 이에 대해 나름대로 정리를 해보았습니다.  
    (참고로 torch에서 말하는 dimension과 numpy에서 나오는 axis는 동일한 의미입니다. 여기에서는 **torch** 기준으로 보겠습니다.)
    
![ch1_2](/assets/img/CH1/ch1_2.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
```
Shape of tensor : torch.Size([dim=0, dim=1, dim=2 ... ])
```

1. 2D Tensor  
![ch1_3](/assets/img/CH1/ch1_3.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
    ```
    Shape of tensor : torch.Size([dim=0, dim=1])
    Shape of tensor : torch.Size([# of Row, # of Column])
    Shape of tensor : torch.Size([2, 3])
    ```

2. 3D Tensor
![ch1_4](/assets/img/CH1/ch1_4.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
    ```
    Shape of tensor : torch.Size([dim=0, dim=1, dim=2])
    Shape of tensor : torch.Size([# of Depth, # of Row, # of Column])
    Shape of tensor : torch.Size([3, 2, 3])
    ```
- 차원이 하나씩 추가될 때, 이 추가된 영역이 앞의 차원에 붙는다는 것에 주의합니다.

<br>  

---
### 3. Handling Tensors - Part 1

- Broadcasting

```python
# For addtion, and subtraction, two matrices' size should be same
# However, in Pytorch, it can automatically fit their size using Broadcasting
# Warning! Be sure with your tensor's size
tensor1 = torch.FloatTensor([[1, 2, 3],[4, 5, 6]]) ## Size : 2 x 3
tensor2 = torch.FloatTensor([[1],[2]]) ## Size : 3 x 1
print("tensor1 : ", tensor1)
print(tensor1.size())
print("--------------------")
print("tensor2 : ", tensor2)
print(tensor2.size())
print("--------------------")
print("Broadcasting: ", tensor1 + tensor2)
```
    [Output]
    tensor1 :  tensor([[1., 2., 3.],
            [4., 5., 6.]])
    torch.Size([2, 3])
    --------------------
    tensor2 :  tensor([[1.],
            [2.]])
    torch.Size([2, 1])
    --------------------
    Broadcasting:  tensor([[2., 3., 4.],
            [6., 7., 8.]])

<br>  

- Matrix Multiplication & Multiplication

```python
tensor1 = torch.FloatTensor([[1, 2],[4, 5]]) ## Size : 2 x 2
tensor2 = torch.FloatTensor([[1],[2]]) ## Size : 2 x 1
print(tensor1.matmul(tensor2)) ## Size : 2 x 1
print(tensor1 * tensor2) ## Size : 2 x 2
print(tensor1.mul(tensor2)) ## This result is same as the result of *
```
    [Output]
    tensor([[ 5.],
            [14.]])
    tensor([[ 1.,  2.],
            [ 8., 10.]])
    tensor([[ 1.,  2.],
            [ 8., 10.]])

<br>  

- (Tip) Rules for Element-wise multiplication
    - 위의 **Broadcasting**이나, **Multiplication**에서 나온 결과를 보면 우리가 아는 행렬의 덧셈, 뺄셈, 곱셈 법칙이 아닌, **Element-wise**한 계산 법칙이 적용되어 자동으로 Size를 맞춰주는 것을 알 수 있습니다. 다음은 이에 따르기 위해 만족되어야 하는 규칙입니다.
    
    1. 각 Tensor는 적어도 한 개 이상의 dimension을 가지고 있어야 합니다.
    2. 각 Tensor들을 비교하여 다음의 조건들 중 하나가 만족되어야 합니다.
        - 모든 dimension size들이 똑같다.
        - dimension size들 중 하나가 1이다.
        - dimension size들 중 하나가 존재하지 않는다.

<br>
 
---
## Reference

- [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788)

- [Understanding dimensions in PyTorch](https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be)

- [Difference between Arrays and Tensors](https://python.plainenglish.io/numpy-arrays-vs-tensorflow-tensors-95a9c39e1c17)