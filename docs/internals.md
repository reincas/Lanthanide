# Internals of the Lanthanide package

Here you find the documentation of some classes and functions, which are not necessary for using the Lanthanide package,
but might be interesting, if you want to dig deeper. You can import each of them from `lanthanide`.

## HalfInt class

The value of quantum numbers of angular momenta in quantum mechanics, in general may be integer or half-integer. The
class `HalfInt` is a lean solution for the representation of half-integer values. To create a `HalfInt` object, provide
its numerator as a parameter. The class supports basic mathematical operations involving other `HalfInt` of `int`
objects. Results with even numerator are converted into ìnt`objects:

```
from lanthanide import HalfInt
print(HalfInt(1) + 2 == HalfInt(5))
print(4 * HalfInt(3) == 6)
print(5 * HalfInt(1) == HalfInt(5))
print(HalfInt(5) >= 1)
print(HalfInt(1) - HalfInt(5) == -2)
print(HalfInt(3) - HalfInt(2) == HalfInt(1))
```

## Matrix class

... to be added ...

## State classes

... to be added ...

