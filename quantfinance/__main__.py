import sys

from quantfinance.tensorflow_quant import fib

if __name__ == "__main__":
    n = int(sys.argv[1])
    print(fib(n))
