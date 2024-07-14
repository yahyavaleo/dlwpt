# Exercises

## 1. Start Python to get an interactive prompt.

### 1.a. What Python version are you using?

My system (Ubuntu 22.04LTS) has three Python environments available:

- the default system-wide environment which has Python 3.10.12 (64-bit),
- the base miniconda environment which has Python 3.10.13 (64-bit), and
- a custom miniconda environment which has Python 3.12.2 (64-bit).

### 1.b. Can you `import torch`? What version of PyTorch do you get?

In my system, PyTorch 2.2.1 is installed in the custom miniconda environment.

```py
import torch
torch.__version__ # Outputs: '2.2.1'
```

### 1.c. What is the result of `torch.cuda.is_available()`? Does it match your expectation based on the hardware you are using?

Calling the function `torch.cuda.is_available()` returns `False`, which is expected since my system does not have any CUDA-capable GPU device.

## 2. Start a Jupyter notebook server.

### 2.a. What version of Python is Jupyter using?

I have installed Jupyter as a standalone desktop application. As such, it has access to all three Python environments available on my system. Since, I have set the custom miniconda environment as the default environment for Jupyter, the Python version is 3.12.2 (64-bit).

### 2.b. Is the location of the `torch` library used by Jupyter the same as the one you imported from the interactive prompt?

Yes, since both the interactive prompt and Jupyter use the same Python environment, they both have access to the same PyTorch installation.

```py
import torch
print(torch) # Outputs: <module 'torch' from '/home/username/miniforge3/envs/envname/lib/python3.12/site-packages/torch/__init__.py>
```
