"""Expression objects (D53): the Python front end of the expression language.

Build an expression with ordinary Python operators; it renders as the
string the case file takes, and the solver compiles that string exactly as
if it had been written in the TOML file::

    from foehncraft.expr import x, y, t, sin, constant, table

    u_max = constant("u_max")
    h = constant("h")
    inflow = 4 * u_max * y * (h - y) / h**2 * table("ramp")(t)
    case["boundary.inlet.velocity"] = [inflow, 0.0, 0.0]
    str(inflow)   # '4 * u_max * y * (h - y) / h ^ 2 * ramp(t)'

Comparisons `<`, `<=`, `>`, `>=` build expressions; equality is `eq(a, b)`
and `ne(a, b)` (Python's `==` stays the identity of the objects). The
logical operators are `&` (and), `|` (or) and `~` (not). Numbers are
written with `repr`, so a float reaches the solver with its bits.

`evaluate(e, ...)` evaluates an expression with the solver's own compiler
and functions (the native module), so Python and the solver agree to the
bit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

Number = int | float

# precedence of the language (higher binds tighter)
_OR, _AND, _NOT, _CMP, _ADD, _MUL, _UNARY, _POW, _ATOM = range(1, 10)


class Expr:
    """An expression of the case-file language."""

    __slots__ = ("text", "prec")

    def __init__(self, text: str, prec: int = _ATOM):
        self.text = text
        self.prec = prec

    # -- rendering
    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Expr({self.text!r})"

    def __toml__(self) -> str:
        return self.text

    def __hash__(self) -> int:
        return hash(self.text)

    def __bool__(self) -> bool:
        raise TypeError(
            "an expression has no truth value in Python; use if_(condition, a, b), "
            "&, | and ~ to build conditions"
        )

    # -- arithmetic
    def __add__(self, o):
        return _binary(self, "+", o, _ADD)

    def __radd__(self, o):
        return _binary(o, "+", self, _ADD)

    def __sub__(self, o):
        return _binary(self, "-", o, _ADD)

    def __rsub__(self, o):
        return _binary(o, "-", self, _ADD)

    def __mul__(self, o):
        return _binary(self, "*", o, _MUL)

    def __rmul__(self, o):
        return _binary(o, "*", self, _MUL)

    def __truediv__(self, o):
        return _binary(self, "/", o, _MUL)

    def __rtruediv__(self, o):
        return _binary(o, "/", self, _MUL)

    def __mod__(self, o):
        return _binary(self, "%", o, _MUL)

    def __rmod__(self, o):
        return _binary(o, "%", self, _MUL)

    def __pow__(self, o):
        return _power(self, o)

    def __rpow__(self, o):
        return _power(o, self)

    def __neg__(self):
        return Expr("-" + _paren(self, _UNARY + 1), _UNARY)

    def __pos__(self):
        return self

    def __abs__(self):
        return _call("abs", self)

    def __floor__(self):
        return _call("floor", self)

    def __ceil__(self):
        return _call("ceil", self)

    def __round__(self, ndigits=None):
        if ndigits is not None:
            raise TypeError("round() of an expression takes no digits")
        return _call("round", self)

    # -- comparisons and logic
    def __lt__(self, o):
        return _binary(self, "<", o, _CMP, chain=False)

    def __le__(self, o):
        return _binary(self, "<=", o, _CMP, chain=False)

    def __gt__(self, o):
        return _binary(self, ">", o, _CMP, chain=False)

    def __ge__(self, o):
        return _binary(self, ">=", o, _CMP, chain=False)

    def __and__(self, o):
        return _binary(self, "and", o, _AND)

    def __rand__(self, o):
        return _binary(o, "and", self, _AND)

    def __or__(self, o):
        return _binary(self, "or", o, _OR)

    def __ror__(self, o):
        return _binary(o, "or", self, _OR)

    def __invert__(self):
        return Expr("not " + _paren(self, _NOT), _NOT)


def _number(v: Number) -> Expr:
    if isinstance(v, bool):
        v = float(v)
    if isinstance(v, int):
        text = str(v)
    else:
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"an expression takes finite numbers only, not {v}")
        text = repr(v)
    return Expr(text, _UNARY if text.startswith("-") else _ATOM)


def wrap(v: Any) -> Expr:
    """`v` as an expression (numbers become constants of the expression)."""
    if isinstance(v, Expr):
        return v
    if isinstance(v, (int, float)) or hasattr(v, "item"):
        return _number(v.item() if hasattr(v, "item") else v)
    raise TypeError(f"not an expression or a number: {v!r}")


def _paren(e: Expr, need: int) -> str:
    """The text of `e`, in parentheses when it binds weaker than `need`."""
    return e.text if e.prec >= need else f"({e.text})"


def _binary(a, op: str, b, prec: int, chain: bool = True) -> Expr:
    a, b = wrap(a), wrap(b)
    # left-associative: the left operand may have the same precedence, the
    # right one must bind tighter; comparisons do not chain at all
    left = _paren(a, prec if chain else prec + 1)
    right = _paren(b, prec + 1)
    return Expr(f"{left} {op} {right}", prec)


def _power(a, b) -> Expr:
    a, b = wrap(a), wrap(b)
    # right-associative and tighter than a unary minus: (-x)^2, x^(-1)
    return Expr(f"{_paren(a, _POW + 1)} ^ {_paren(b, _POW)}", _POW)


def _call(name: str, *args) -> Expr:
    return Expr(f"{name}(" + ", ".join(wrap(a).text for a in args) + ")", _ATOM)


#: The coordinates of the face or cell centroid (m) and the simulated time (s).
x = Expr("x")
y = Expr("y")
z = Expr("z")
t = Expr("t")
#: The constants of the language.
pi = Expr("pi")
e = Expr("e")

VARIABLES = ("x", "y", "z", "t")
FUNCTIONS1 = (
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "exp", "log", "log10", "sqrt", "abs", "floor", "ceil", "round", "sign",
    "erf", "erfc",
)  # fmt: skip
FUNCTIONS2 = ("atan2", "pow", "min", "max", "hypot", "step")
#: Names of solution fields that later stages bring into expressions.
FIELDS = ("u", "v", "w", "p", "T", "rho", "mu", "k", "omega", "epsilon", "nut", "Y")
KEYWORDS = ("and", "or", "not", "if", "clamp")
RESERVED = frozenset(VARIABLES + ("pi", "e") + FUNCTIONS1 + FUNCTIONS2 + FIELDS + KEYWORDS)


def constant(name: str) -> Expr:
    """A constant of the case's `[constants]` by name."""
    _check_name(name)
    return Expr(name)


def _check_name(name: str) -> None:
    if not name.isidentifier():
        raise ValueError(f"`{name}` is not a valid name")
    if name in RESERVED:
        raise ValueError(f"`{name}` is a reserved name of the expressions")


class table:
    """A table of the case's `[tables]` by name, called with its argument."""

    def __init__(self, name: str):
        _check_name(name)
        self.name = name

    def __call__(self, arg) -> Expr:
        return _call(self.name, arg)


def sin(a):
    return _call("sin", a)


def cos(a):
    return _call("cos", a)


def tan(a):
    return _call("tan", a)


def asin(a):
    return _call("asin", a)


def acos(a):
    return _call("acos", a)


def atan(a):
    return _call("atan", a)


def sinh(a):
    return _call("sinh", a)


def cosh(a):
    return _call("cosh", a)


def tanh(a):
    return _call("tanh", a)


def exp(a):
    return _call("exp", a)


def log(a):
    """The natural logarithm."""
    return _call("log", a)


def log10(a):
    return _call("log10", a)


def sqrt(a):
    return _call("sqrt", a)


def sign(a):
    return _call("sign", a)


def erf(a):
    """The error function."""
    return _call("erf", a)


def erfc(a):
    """The complementary error function 1 - erf."""
    return _call("erfc", a)


def floor(a):
    return _call("floor", a)


def ceil(a):
    return _call("ceil", a)


def atan2(a, b):
    return _call("atan2", a, b)


def pow(a, b):  # noqa: A001 - the name of the language
    return _call("pow", a, b)


def min_(a, b):
    """`min(a, b)` of the language."""
    return _call("min", a, b)


def max_(a, b):
    """`max(a, b)` of the language."""
    return _call("max", a, b)


def hypot(a, b):
    return _call("hypot", a, b)


def step(a, b):
    """1 where a >= b, else 0."""
    return _call("step", a, b)


def clamp(a, lo, hi):
    return _call("clamp", a, lo, hi)


def if_(condition, a, b):
    """`a` where `condition` holds (is not 0), else `b`."""
    return _call("if", condition, a, b)


def eq(a, b) -> Expr:
    """`a == b` of the language (1 or 0)."""
    return _binary(a, "==", b, _CMP, chain=False)


def ne(a, b) -> Expr:
    """`a != b` of the language (1 or 0)."""
    return _binary(a, "!=", b, _CMP, chain=False)


def _env_text(constants: dict | None, tables: dict | None) -> str:
    from . import _toml

    d: dict[str, Any] = {}
    if constants:
        d["constants"] = dict(constants)
    if tables:
        d["tables"] = dict(tables)
    return _toml.dumps(d)


def evaluate(
    expression,
    points: Iterable[Sequence[float]] | None = None,
    *,
    constants: dict | None = None,
    tables: dict | None = None,
    allowed: str = "xyzt",
    base=None,
    **at: float,
) -> list[float] | float:
    """Evaluates `expression` (an `Expr` or a string) with the solver.

    Either `points`, a sequence of (x, y, z, t), giving a list, or the
    keywords `x=`, `y=`, `z=`, `t=` (default 0), giving one number.
    `constants` and `tables` are as in the case file. Needs the native
    module.
    """
    from ._native import expression as _expression

    single = points is None
    pts = [[at.get("x", 0.0), at.get("y", 0.0), at.get("z", 0.0), at.get("t", 0.0)]]
    if not single:
        pts = [[float(v) for v in p] for p in points]
    values, _ = _expression(
        str(wrap(expression)) if not isinstance(expression, str) else expression,
        _env_text(constants, tables),
        allowed,
        pts,
        None if base is None else str(base),
    )
    return values[0] if single else values


def uses(expression, *, constants: dict | None = None, tables: dict | None = None) -> str:
    """The variables (`x`, `y`, `z`, `t`) that `expression` reads, after
    folding the constants (native module)."""
    from ._native import expression as _expression

    src = expression if isinstance(expression, str) else str(wrap(expression))
    _, u = _expression(src, _env_text(constants, tables), "xyzt", [], None)
    return u
