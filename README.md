This code is basically a direct translation of the C++ CellTree2D by Jay Hennen,
at: 

https://github.com/NOAA-ORR-ERD/cell_tree2d/blob/master/src/cell_tree2d.cpp

(Public Domain)

It implements the cell tree as described in:

Garth, C., & Joy, K. I. (2010). Fast, memory-efficient cell location in
unstructured grids for visualization. IEEE Transactions on Visualization and
Computer Graphics, 16(6), 1541-1550.

Available at: https://escholarship.org/content/qt0vq7q87f/qt0vq7q87f.pdf

The main benefit is ease of development and distribution. Numba integrates
seemlessly with Python, and distribution requires only sharing of Python source
files.

The query methods are implemented using a stack rather than recursion: numba
does not seem to able to optimize recursion as efficiently as the C++ compilers
(in this case). Tree building speed is basically the same, although this
currently pre-allocates pessimistically. Serial queries are ~30% slower, but
numba allows parallellization via a single keyword, in which case queries are
faster -- the C++ implementation is exclusively serial (of course it could be
parallelized fairly easily with OpenMP).

Documentation [here](https://huite.github.io/numba_celltree/)
