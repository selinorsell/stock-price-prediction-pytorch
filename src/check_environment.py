import sys

import matplotlib
import numpy
import pandas
import sklearn
import torch


def main() -> None:
    print("Python:", sys.version)
    print("NumPy:", numpy.__version__)
    print("Pandas:", pandas.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU (this project can still run on CPU)")


if __name__ == "__main__":
    main()
