```python
>>> def factorial(n):
  # n! = n * (n-1) * (n-2) * ... * 1
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
```
