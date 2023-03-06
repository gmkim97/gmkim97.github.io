---
title: PyTorch Learning_Tensor Manipulation (1)
author: GMKim
date: 2022-08-13 11:45:00 +0900
categories: [Machine Learning, PyTorch]
tags: [Machine Learning, PyTorch, Tensor]
---

## Introduction

This note is based on the lecture note (to be exact, the wiki documentary) written in Korean.  
It is about the introductory informations of Deep Learning with PyTorch.
- [PyTorch로 시작하는 딥 러닝 입문(KOR)](https://wikidocs.net/book/2788){:target="_blank"}

Here, I'd like to write **what is Tensor**, **how can we allocate it**, and **several functions to deal with**.
<br>

---
## Tensor Manipulation - Part 1

### 1. What's Tensor?

- Definition
    - First, Wikipedia defines the Tensor as follows :  
    ```
    In mathematics, a tensor is an algebraic object that describes a multilinear relationship between sets of algebraic objects related to a vector space.
    ```
    Since it sounds quite difficult and complicated, let's make it easier and simple.
    ```
    A tensor is the tool to represent a multi-dimensional arrays.
    ``` 
    ![ch1_1](/assets/img/CH1/ch1_1.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

<br>  

- 1D with Numpy
    - To represent a multi-dimensional array, we usually use numpy library at Python. As a simple review, let's write 1-D array using numpy array.  

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
    - Next, let's write 2-D array using numpy array.  

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
    - Then, why do we need to use tensor from Pytorch instead of numpy array?  
    I searched for the reason and found the post, `What is the Difference Between NumPy Arrays and Tensorflow Tensors?` written by **Konstantinos Giorgas**. Of course, we are going to use Pytorch Tensor instead of Tensorflow, anyway...  
    - To summarize it,
        1. Tensor is able to perform GPU Acceleration.
        2. Tensor can differentiate automatically.
        3. Tensor can handle various data type including not only the number but also the string.
    - Thus, using Tensor is more appropriate for Deep Learning which operates heavier than simple Machine Learniing.

<br>  

---
### 2. Tensor Allocation

- 1D with Pytorch
    - This time, let's write 1-D array using torch library from Pytorch.  
    Note that the output will be tensor type, not array type.  

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
    - Let's write 2-D array with Pytorch tensor.  

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
    - Whether it's numpy array or torch tensor, it is very important to figure out the **size of Tensor** for using methods or functions. Here, I try to organize it in my words.
    (Note that the dimension of torch and axis of numpy share similar meaning. And this explanation is based on **torch**)
    
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
- Notice that the dimension is added in front of existing dimensions. 

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
    - From the results of **Broadcasting** or **Multiplication**, we can recognize that it automatically fits the size because of the **Element-wise** operation, not the matrix operations. These are the rules to satisfy.

    1. Each Tensor should have at least one dimension.
    2. One of the following rules is satisfied between two Tensors.
        - Both sizes of dimension are identical.
        - One of the size should be 1.
        - One of the size should not exist.
<br>
 
---
## Reference

- [PyTorch로 시작하는 딥 러닝 입문(KOR)](https://wikidocs.net/book/2788){:target="_blank"}

- [Understanding dimensions in PyTorch](https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be){:target="_blank"}

- [Difference between Arrays and Tensors](https://python.plainenglish.io/numpy-arrays-vs-tensorflow-tensors-95a9c39e1c17){:target="_blank"}