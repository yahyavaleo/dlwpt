# Exercises

## 1. Take several pictures of red, blue, and green items with your phone or other digital camera (or download some from the internet, if a camera is not available).

### 1.a. Load each image, and convert it to a tensor.

To load the image, we use the `imageio.imread()` method. The resulting numpy array is converted to a tensor using the `torch.from_numpy()` method. However, since `imageio.imread()` outputs an array of shape $H \times W \times C$ but we need a tensor of shape $C \times H \times W$, we use the `torch.permute()` method. Finally, we convert the tensor to float data type.

```py
import torch
import imageio.v2 as imageio

red_fruit_arr = imageio.imread("data/fruits/red-fruit.jpg")
green_fruit_arr = imageio.imread("data/fruits/green-fruit.jpg")
blue_fruit_arr = imageio.imread("data/fruits/blue-fruit.jpg")

red_fruit = torch.from_numpy(red_fruit_arr)
green_fruit = torch.from_numpy(green_fruit_arr)
blue_fruit = torch.from_numpy(blue_fruit_arr)

red_fruit = torch.permute(red_fruit, (2, 0, 1))
green_fruit = torch.permute(green_fruit, (2, 0, 1))
blue_fruit = torch.permute(blue_fruit, (2, 0, 1))

red_fruit = red_fruit.float()
green_fruit = green_fruit.float()
blue_fruit = blue_fruit.float()
```

### 1.b. For each image tensor, use the `mean()` method to get a sense of how bright the image is.

Simply using the `torch.mean()` method over the image tensor will produce a single mean value over all channels. This value gives an indication of how bright the image is.

```py
torch.mean(red_fruit).item(), torch.mean(green_fruit).item(), torch.mean(blue_fruit).item()
```

**Output**

```txt
(93.08065032958984, 81.8938217163086, 95.56259155273438)
```

### 1.c. Take the mean of each channel of your images. Can you identify the red, green, and blue items from only the channel averages?

We call the `torch.mean()` method over each channel for each image. The mean value of the red channel is highest for the red fruit image, the mean value of the green channel is highest for the green fruit image, and the mean value of the blue channel is highest for the blue fruit image.

```py
red = torch.mean(red_fruit[0, :, :]).item()
green = torch.mean(red_fruit[1, :, :]).item()
blue = torch.mean(red_fruit[2, :, :]).item()

print(f"Red: {red:4.2f} - Green: {green:4.2f} - Blue: {blue:4.2f}")
```

**Output**

```txt
Red: 184.42 - Green: 42.02 - Blue: 52.80
```

```py
red = torch.mean(green_fruit[0, :, :]).item()
green = torch.mean(green_fruit[1, :, :]).item()
blue = torch.mean(green_fruit[2, :, :]).item()

print(f"Red: {red:4.2f} - Green: {green:4.2f} - Blue: {blue:4.2f}")
```

**Output**

```txt
Red: 95.72 - Green: 134.20 - Blue: 15.76
```

```py
red = torch.mean(blue_fruit[0, :, :]).item()
green = torch.mean(blue_fruit[1, :, :]).item()
blue = torch.mean(blue_fruit[2, :, :]).item()

print(f"Red: {red:4.2f} - Green: {green:4.2f} - Blue: {blue:4.2f}")
```

**Output**

```txt
Red: 66.75 - Green: 93.25 - Blue: 126.69
```

## 2. Select a relatively large file containing Python source code.

### 2.a. Build an index of all the words in the source file (feel free to make your tokenization as simple or as complex as you like; we suggest starting with replacing `r"[^a-zA-Z0-9_]+"` with spaces).

In order to tokenize the source file into alphabetical words, we use a regular expression to filter out anything that is not a word composed of only alphabetical letters. Then we created a sorted set of these tokens to remove duplicates. Finally, we create an index of these tokens using this sorted set.

```py
import re
import torch

with open("data/python/main.py", mode="r", encoding="utf8") as f:
    text = f.read()

def get_tokens(input_str):
    input_str = input_str.lower()
    tokens = re.findall(r'\b[a-zA-Z]+\b', input_str)
    return tokens

tokens = sorted(set(get_tokens(text)))
token2index_dict = {token: i for (i, token) in enumerate(tokens)}
len(token2index_dict) # 146
```

### 2.b. Compare your index with the one we made for _Pride and Prejudice_. Which is larger?

This index file accounts for only $146$ words whereas the one for _Pride and Prejudice_ has $7261$ words. This is because, the Python source code has only a very limited vocabulary whereas _Pride and Prejudice_ uses a large chunk of English vocabulary.

### 2.c. Create the one-hot encoding for the source file.

The one-hot encoding for the source file is created by initializing a zero-filled tensor with as many rows as there are words in the text and as many columns as there are unique tokens in the text. We iterate over each token, find its index using the `token2index_dict` and one-hot encode the corresponding position in the tensor.

```py
tokens = get_tokens(text)
token_t = torch.zeros(len(tokens), len(token2index_dict))

for i, token in enumerate(tokens):
    token_index = token2index_dict[token]
    token_t[i][token_index] = 1

print(token_t.shape) # torch.Size([2232, 146])
```

### 2.d. What information is lost with this encoding? How does this information compare to what is lost in the _Pride and Prejudice_ encoding?

One-hot encoding represents each token independently of its position in the sequence. This means that the sequential information, which is crucial for understanding the syntax and semantics of source code is lost. For example, the difference between `if condition` and `condition if` is not captured. This loss of order information does not strictly apply to the _Pride and Prejudice_ since is is an English text.
