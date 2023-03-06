---
title: PyTorch Learning_Tensor Manipulation (2)
author: GMKim
date: 2022-08-13 11:45:00 +0900
categories: [Machine Learning, PyTorch]
tags: [Machine Learning, PyTorch, Tensor]
---

## Introduction

This note is based on the lecture note (to be exact, the wiki documentary) written in Korean.  
It is about the introductory informations of Deep Learning with PyTorch.
- [PyTorch로 시작하는 딥 러닝 입문(KOR)](https://wikidocs.net/book/2788){:target="_blank"}
  
Here, I'd like to write more about some functions to handle Pytorch Tensor.
<br>

---
## Tensor Manipulation - Part 2

### 3. Handling Tensors - Part 2

- Mean
    - It gives the mean value of Tensor components. You can designate specific row or column using dimension parameter(dim).

```python
tensor1 = torch.FloatTensor([1, 2]) # 1D Size : 1 x 2
tensor2 = torch.FloatTensor([[1, 2],[3, 4]]) # 2D Size : 2 x 2
print(tensor1)
print(tensor2)
print("--------------------")
print(tensor1.mean()) ## mean value for total components
print(tensor1.mean().size()) ## mean value of tensor is also tensor
print("--------------------")
print(tensor2.mean())
print(tensor2.mean().size())
print("--------------------")
print(tensor2.mean(dim=0)) 
print(tensor2.mean(dim=1)) 
print(tensor2.mean(dim=-1))
```
    [Output]
    tensor([1., 2.])
    tensor([[1., 2.],
            [3., 4.]])
    --------------------
    tensor(1.5000)
    torch.Size([])
    --------------------
    tensor(2.5000)
    torch.Size([])
    --------------------
    tensor([2., 3.])
    tensor([1.5000, 3.5000])
    tensor([1.5000, 3.5000])

<br>  

- Sum
    - It gives the total sum of Tensor components. You can also designate specific row or column using dimension parameter(dim).

```python
tensor1 = torch.FloatTensor([1, 2]) # 1D Size : 1 x 2
tensor2 = torch.FloatTensor([[1, 2],[3, 4]]) # 2D Size : 2 x 2
print(tensor1)
print(tensor2)
print("--------------------")
print(tensor1.sum()) ## sum value for total components
print(tensor1.sum().size()) ## sum value of tensor is also tensor
print("--------------------")
print(tensor2.sum())
print(tensor2.sum().size())
print("--------------------")
print(tensor2.sum(dim=0)) 
print(tensor2.sum(axis=0))
print(tensor2.sum(dim=1)) 
print(tensor2.sum(dim=-1))
```
    [Output]
    tensor([1., 2.])
    tensor([[1., 2.],
            [3., 4.]])
    --------------------
    tensor(3.)
    torch.Size([])
    --------------------
    tensor(10.)
    torch.Size([])
    --------------------
    tensor([4., 6.])
    tensor([4., 6.])
    tensor([3., 7.])
    tensor([3., 7.])

<br>  

- (Tip) How to sum / mean at specific dimension?
    - Here, I'm going to show you the process how it calculates when designating dimension parameter(dim).
    - Examples are brought from previous `Size of Tensors` chapter.
    - Keypoint : **Collapse the 'dimension(axis)'**
    1. 2D Tensor  
    ![ch1_3](/assets/img/CH1/ch1_3.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
    ![ch1_5](/assets/img/CH1/ch1_5.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

        ```python
        t2 = torch.FloatTensor([[0, 1, 2],[3, 4, 5]])
        print(t2.shape)
        print(t2)
        print("--------------------")
        print("[dim=0]")
        print(t2.sum(dim=0).shape)
        print(t2.sum(dim=0))
        print("--------------------")
        print("[dim=1]")
        print(t2.sum(dim=1).shape)
        print(t2.sum(dim=1))
        ```
            [Output]
            torch.Size([2, 3])
            tensor([[0., 1., 2.],
                    [3., 4., 5.]])
            --------------------
            [dim=0]
            torch.Size([3])
            tensor([3., 5., 7.])
            --------------------
            [dim=1]
            torch.Size([2])
            tensor([ 3., 12.])    

    2. 3D Tensor  
    ![ch1_4](/assets/img/CH1/ch1_4.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
    ![ch1_6](/assets/img/CH1/ch1_6.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

        ```python
        t3 = torch.FloatTensor([[[0, 1, 2],[3, 4, 5]],[[0, 1, 2],[3, 4, 5]],[[0, 1, 2],[3, 4, 5]]])
        print(t3.shape)
        print(t3)
        print("--------------------")
        print("[dim=0]")
        print(t3.sum(dim=0).shape)
        print(t3.sum(dim=0))
        print("--------------------")
        print("[dim=1]")
        print(t3.sum(dim=1).shape)
        print(t3.sum(dim=1))
        print("--------------------")
        print("[dim=2]")
        print(t3.sum(dim=2).shape)
        print(t3.sum(dim=2))
        ```
            [Output]
            torch.Size([3, 2, 3])
            tensor([[[0., 1., 2.],
                    [3., 4., 5.]],
            
                    [[0., 1., 2.],
                    [3., 4., 5.]],
            
                    [[0., 1., 2.],
                    [3., 4., 5.]]])
            --------------------
            [dim=0]
            torch.Size([2, 3])
            tensor([[ 0.,  3.,  6.],
                    [ 9., 12., 15.]])
            --------------------
            [dim=1]
            torch.Size([3, 3])
            tensor([[3., 5., 7.],
                    [3., 5., 7.],
                    [3., 5., 7.]])
            --------------------
            [dim=2]
            torch.Size([3, 2])
            tensor([[ 3., 12.],
                    [ 3., 12.],
                    [ 3., 12.]])

<br>  

- Max & Argmax

```python
t = torch.FloatTensor([[1, 2],[3, 4]])
print(t)
print("--------------------")
print(t.max())
print("--------------------")
print(t.max(dim=0))
print(t.max(dim=0)[0]) ## Values
print(t.max(dim=0)[1]) ## Indices
print("--------------------")
print(t.argmax())
print(t.argmax(dim=0)) ## Result is same as the one from t.max(dim=0)[1]
```
    [Output]
    tensor([[1., 2.],
            [3., 4.]])
    --------------------
    tensor(4.)
    --------------------
    torch.return_types.max(
    values=tensor([3., 4.]),
    indices=tensor([1, 1]))
    tensor([3., 4.])
    tensor([1, 1])
    --------------------
    tensor(3)
    tensor([1, 1])

<br>  

- View
    - It reshapes the size of Tensor in condition that the total size is maintained.

```python
t3 = torch.FloatTensor([[[0, 1, 2],[3, 4, 5]],[[6, 7, 8],[9, 10, 11]]])
print(t3)
```
    [Output]
    tensor([[[ 0.,  1.,  2.],
             [ 3.,  4.,  5.]],
    
            [[ 6.,  7.,  8.],
             [ 9., 10., 11.]]])



```python
# Both output size of Tensor
print(t3.size())
print(t3.shape)
```
    [Output]
    torch.Size([2, 2, 3])
    torch.Size([2, 2, 3])



```python
# Reshape into 2D Tensor
print(t3.view([-1,3])) ## -1 means reshape automatically (Leave it to Pytorch)
print(t3.view([2,6])) ## Shape must be maintained (2x2x3)=(2x6)=12
```
    [Output]
    tensor([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.]])
    tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
            [ 6.,  7.,  8.,  9., 10., 11.]])



```python
# Reshape into 3D Tensor
print(t3.view([-1,2,3])) ## -1 means reshape automatically (Leave it to Pytorch)
print(t3.view([2,3,2])) ## Shape must be maintained (2x2x3)=(2x3x2)=12
```
    [Output]
    tensor([[[ 0.,  1.,  2.],
             [ 3.,  4.,  5.]],
    
            [[ 6.,  7.,  8.],
             [ 9., 10., 11.]]])
    tensor([[[ 0.,  1.],
             [ 2.,  3.],
             [ 4.,  5.]],
    
            [[ 6.,  7.],
             [ 8.,  9.],
             [10., 11.]]])
  
<br>  

- Squeeze & Unsqueeze
    - Squeeze : It removes the dimension of value 1, and the total dimension of Tensor **decreases**.
    - Unsqueeze : It adds the dimension of value 1 at specific location, and the total dimension of Tensor **increases**.

```python
t2 = torch.FloatTensor([[0],[1],[2]])
print(t2)
print(t2.size())
```
    [Output]
    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])


```python
# Squeeze
# Eliminate the dimension with 1 == Reduce total dimensions
print(t2.squeeze())
print(t2.squeeze().size())
```
    [Output]
    tensor([0., 1., 2.])
    torch.Size([3])



```python
# Let t1 be the output of t2.squeeze()
t1 = t2.squeeze()
```

```python
# Unsqueeze
# Add the dimension with 1 at specific place == increase total dimensions
print(t1.unsqueeze(dim=0).size())
print(t1.unsqueeze(dim=0))
print("--------------------------")
print(t1.unsqueeze(dim=1).size())
print(t1.unsqueeze(dim=1))
```
    [Output]
    torch.Size([1, 3])
    tensor([[0., 1., 2.]])
    --------------------------
    torch.Size([3, 1])
    tensor([[0.],
            [1.],
            [2.]])
  
<br>  

- Concatenate & Stack
    - Concatenate : It concatenates additional Tensor **maintaining the total dimension.**
    - Stack : It stacks additional Tensor **increasing the total dimension** (Ex. dim 2 -> 3)

```python
# Concatenate
## Non-empty tensors provided must have the same shape, except in the cat dimension
x = torch.randint(high=10, size=(2,2)) ## 2 x 2
y = torch.randint(high=10, size=(2,2)) ## 2 x 2
print(x)
print(y)
```
    [Output]
    tensor([[7, 8],
            [3, 8]])
    tensor([[5, 6],
            [5, 2]])



```python
print(torch.cat([x,y])) # This result should be same as dim=0
print(torch.cat([x,y], dim=0)) ## 4 x 2
print(torch.cat([x,y], dim=1)) ## 2 x 4
```
    [Output]
    tensor([[7, 8],
            [3, 8],
            [5, 6],
            [5, 2]])
    tensor([[7, 8],
            [3, 8],
            [5, 6],
            [5, 2]])
    tensor([[7, 8, 5, 6],
            [3, 8, 5, 2]])



```python
# Stack == concatenate + unsqueeze
## Concatenate a sequence of tensors along a new dimension
## All tensors need to be of the same size
x = torch.randint(high=10, size=(2,2)) ## 2 x 2
y = torch.randint(high=10, size=(2,2)) ## 2 x 2
z = torch.randint(high=10, size=(2,2)) ## 2 x 2
print(x)
print(y)
print(z)
```
    [Output]
    tensor([[7, 8],
            [7, 6]])
    tensor([[4, 3],
            [8, 8]])
    tensor([[1, 6],
            [3, 7]])

![ch1_7](/assets/img/CH1/ch1_7.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

```python
print(torch.stack([x,y,z])) # This result should be same as dim=0
print(torch.stack([x,y,z], dim=0)) ## 3 x 2 x 2
print(torch.stack([x,y,z], dim=1)) ## 2 x 3 x 2
print(torch.stack([x,y,z], dim=2)) ## 2 x 2 x 3
```
    [Output]
    tensor([[[7, 8],
             [7, 6]],
    
            [[4, 3],
             [8, 8]],
    
            [[1, 6],
             [3, 7]]])
    tensor([[[7, 8],
             [7, 6]],
    
            [[4, 3],
             [8, 8]],
    
            [[1, 6],
             [3, 7]]])
    tensor([[[7, 8],
             [4, 3],
             [1, 6]],
    
            [[7, 6],
             [8, 8],
             [3, 7]]])
    tensor([[[7, 4, 1],
             [8, 3, 6]],
    
            [[7, 8, 3],
             [6, 8, 7]]])

<br>  

- ones_like & zeros_like
    - **_like** means that **it will have the same dimension with defined Tensor**. Thus, it gives the result of ones or zeros with designated size.

```python
x = torch.randint(high=10, size=(2,3))
print(x)
```
    [Output]
    tensor([[7, 4, 8],
            [6, 8, 6]])



```python
# ones_like
## _like means same size with tensor x
t1 = torch.ones_like(x)
print(t1)
# ones
t1_1 = torch.ones(size=(2,3))
print(t1_1)
```
    [Output]
    tensor([[1, 1, 1],
            [1, 1, 1]])
    tensor([[1., 1., 1.],
            [1., 1., 1.]])



```python
# zeros_like
## _like means same size with tensor x
t0 = torch.zeros_like(x)
print(t0)
# zeros
t0_1 = torch.zeros(size=(2,3))
print(t0_1)
```
    [Output]
    tensor([[0, 0, 0],
            [0, 0, 0]])
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

<br>  

- In-place operation
    - If you apply In-place operation at the end of each function, you can save or accumulate ongoing results at first defined variable.

```python
x = torch.FloatTensor([[1, 2],[3, 4]])
print(x)
```
    [Output]
    tensor([[1., 2.],
            [3., 4.]])



```python
print(x.mul(2))
print(x)
```
    [Output]
    tensor([[2., 4.],
            [6., 8.]])
    tensor([[1., 2.],
            [3., 4.]])



```python
## In-place operation saves its result into variable
print(x.mul_(2))
print(x)
```
    [Output]
    tensor([[2., 4.],
            [6., 8.]])
    tensor([[2., 4.],
            [6., 8.]])

<br>  

---
## Reference

- [PyTorch로 시작하는 딥 러닝 입문(KOR)](https://wikidocs.net/book/2788){:target="_blank"}
- [Understanding dimensions in PyTorch](https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be){:target="_blank"}