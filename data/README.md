# Data

`MNIST/raw/` holds the MNIST test set (10,000 images and labels, IDX format, gzip): `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz`.

- Source: Y. LeCun, C. Cortes and C. J. C. Burges, The MNIST database of handwritten digits.
- Reference: Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.
- License: Yann LeCun and Corinna Cortes hold the copyright of MNIST, a derivative work of the original NIST datasets. It is made available under the Creative Commons Attribution-Share Alike 3.0 license. The files are unmodified.

The code reads these files directly; torchvision is not needed. `scripts/verify_dataset.py --dataset mnist --root ./data` checks that they load.
