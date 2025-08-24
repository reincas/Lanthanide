# Internals of the Lanthanide package

Here you find the documentation of some classes and functions, which are not necessary for using the Lanthanide package,
but might be interesting, if you want to dig deeper. You can import each of them from `lanthanide`.

## HalfInt class

The value of quantum numbers of angular momenta in quantum mechanics in general may be integer or half-integer. The
class `HalfInt` is a lean solution for the representation of half-integer values. To create a `HalfInt` object, provide
its numerator as a parameter. The class supports basic mathematical operations involving other `HalfInt` of `int`
objects. Results with even numerator are converted into `ìnt` objects:

```
from lanthanide import HalfInt

print(HalfInt(1) + 2 == HalfInt(5))
print(4 * HalfInt(3) == 6)
print(5 * HalfInt(1) == HalfInt(5))
print(HalfInt(5) >= 1)
print(HalfInt(1) - HalfInt(5) == -2)
print(HalfInt(3) - HalfInt(2) == HalfInt(1))
```

## Wigner symbols

The Lanthanide package uses Wigner 3-j symbols extensively to calculate matrix elements of tensor operators. To get the
value of a 3-j symbol

$$
\begin{pmatrix}
j_1 & j_2 & j_3 \\
m_1 & m_2 & m_3
\end{pmatrix}
$$

you use the function `wigner3j(j1, j2, j3, m1, m2, m3)`. The arguments may be either `int` or `HalfInt` objects:

```
from lanthanide import wigner3j
factor = wigner3j(2, HalfInt(3), HalfInt(3), 0, -HalfInt(1), HalfInt(1))
```

The Lanthanide package does not require Wigner 6-j symbols, because it builds tensor matrices in the space of
determinantal product states. However, if you want to process the matrices in higher-order coupling, you might be
interested in Wigner 6-j symbols

$$
\begin{Bmatrix}
j_1 & j_2 & j_3 \\
l_1 & l_2 & l_3
\end{Bmatrix}
$$

as well. For that purpose, the package provides the function `wigner6j(j1, j2, j3, l1, l2, l3)`. The arguments may be
either `int` or `HalfInt` objects again:

```
from lanthanide import wigner6j
factor = wigner6j(1, 2, 3, 2, 1, 2)
```

## Matrix class

... to be added ...

## State classes

... to be added ...

