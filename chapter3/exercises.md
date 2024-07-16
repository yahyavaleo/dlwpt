# Exercises

## 1. Create a tensor `a` from `list(range(9))`. Predict and then check the size, offset, and stride.

The tensor `a` can be created using a call to `torch.tensor()` with `list(range(9))` as argument. Since it is a one-dimensional tensor with a total length of 9 elements, the size is `[9,]`. The tensor `a` is not a view of any other tensor, so the offset is `0`. Finally, since there is only row with 9 columns, the stride is `(1,)`.

```py
a = torch.tensor(list(range(9)))
print(f"a.size():           {a.size()}")
print(f"a.storage_offset(): {a.storage_offset()}")
print(f"a.stride():         {a.stride()}")
```

**Output**

```txt
a.size():           torch.Size([9])
a.storage_offset(): 0
a.stride():         (1,)
```

### 1.a. Create a new tensor using `b = a.view(3, 3)`. What does `view` do? Check that `a` and `b` share the same storage.

The function `view()` returns a new tensor with the same data as the original tensor but with a different shape. However, the returned tensor must have the same number of elements.

```py
b = a.view(3, 3)
id(a.untyped_storage()) == id(b.untyped_storage())
```

**Output**

```txt
True
```

### 1.b. Create a tensor `c = b[1:, 1:]`. Predict and then check the size, offset, and stride.

The tensor `c` is a view of the tensor `b` but only selects the second and third row and second and third column, it has a size of `[2, 2]`. Because it skips the first row entirely (of 3 elements) and skips the first column of the second row, it has an offset of `4`. Finally, the next row is accessed by moving past 3 elements and columns are adjacent, the stride is `(3, 1)`.

```py
c = b[1:, 1:]
print(f"c.size():           {c.size()}")
print(f"c.storage_offset(): {c.storage_offset()}")
print(f"c.stride():         {c.stride()}")
```

**Output**

```txt
c.size():           torch.Size([2, 2])
c.storage_offset(): 4
c.stride():         (3, 1)
```

## 2. Pick a mathematical operation like cosine or square root. Can you find a corresponding function in the `torch` library?

### 2.a. Apply the function element-wise to `a`. Why does it return an error?

The following code does not return an error in newer versions of PyTorch. However, in older versions, these element-wise functions only operated on tensors with data type of floating-point and because `a` was created from `list(range(9))`, it has a data type of integers. Therefore, this code would produce an error in older versions of PyTorch.

```py
sqrt_a = torch.sqrt(a)
```

### 2.b. What operation is required to make the function work?

In older versions of PyTorch, the tensor `a` need to be converted to floating-point type.

```py
a = a.float()
```

### 2.c. Is there a version of your function that operates in place?

Yes, there is a corresponding tensor method with a trailing underscore.

```py
a.sqrt_()
```
