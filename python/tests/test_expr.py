"""Expression objects (D53): rendering, precedence and the solver's values."""

import math
import random

import pytest
from conftest import needs_native

from foehncraft import expr as E
from foehncraft.expr import constant, eq, if_, ne, pi, t, table, x, y, z


def test_rendering():
    assert str(4 * constant("um") * y * (1 - y)) == "4 * um * y * (1 - y)"
    assert str(-(x**2)) == "-x ^ 2"
    assert str((-x) ** 2) == "(-x) ^ 2"
    assert str(x**y**z) == "x ^ y ^ z"
    assert str((x**y) ** z) == "(x ^ y) ^ z"
    assert str(x - (y - z)) == "x - (y - z)"
    assert str((x - y) - z) == "x - y - z"
    assert str(x / (y * z)) == "x / (y * z)"
    assert str((x < 1) & ~(y > 2)) == "x < 1 and not y > 2"
    assert str(if_(eq(x, 0), 1.5, E.sin(pi * t))) == "if(x == 0, 1.5, sin(pi * t))"
    assert str(table("ramp")(t)) == "ramp(t)"
    assert str(0.1 * x) == "0.1 * x"


def test_names_and_truth():
    with pytest.raises(ValueError):
        constant("u")  # a field name, reserved
    with pytest.raises(ValueError):
        constant("sin")
    with pytest.raises(TypeError):
        bool(x < 1)
    with pytest.raises(ValueError):
        E.wrap(float("inf"))


@needs_native
def test_values_of_the_solver():
    rnd = random.Random(1)
    pts = [
        [rnd.uniform(-2, 2), rnd.uniform(0.1, 3), rnd.uniform(-1, 1), rnd.uniform(0, 5)]
        for _ in range(200)
    ]
    cases = [
        (
            x * y - z / y + t % 1.5,
            lambda a, b, c, d: a * b - c / b + (d - 1.5 * math.floor(d / 1.5)),
        ),
        (-(x**2) + (2 * y) ** 0.5, lambda a, b, c, d: -(a**2) + math.sqrt(2 * b)),
        (
            E.sin(pi * x) * E.exp(-t) + E.atan2(z, y),
            lambda a, b, c, d: math.sin(math.pi * a) * math.exp(-d) + math.atan2(c, b),
        ),
        (
            E.log(y) + E.log10(y) + E.sqrt(y),
            lambda a, b, c, d: math.log(b) + math.log10(b) + math.sqrt(b),
        ),
        (
            E.max_(x, z) - E.min_(x, z) + E.hypot(x, z),
            lambda a, b, c, d: max(a, c) - min(a, c) + math.hypot(a, c),
        ),
        (
            if_((x > 0) | (z < 0), E.tanh(x), E.clamp(z, -0.5, 0.5)),
            lambda a, b, c, d: math.tanh(a) if (a > 0 or c < 0) else min(max(c, -0.5), 0.5),
        ),
        (
            E.erf(x) + E.erfc(y),
            lambda a, b, c, d: math.erf(a) + math.erfc(b),
        ),
        (
            E.step(x, 0.25) + abs(z) + E.sign(z),
            lambda a, b, c, d: (1.0 if a >= 0.25 else 0.0) + abs(c) + math.copysign(1.0, c),
        ),
    ]
    for e, f in cases:
        got = E.evaluate(e, pts)
        for p, g in zip(pts, got):
            want = f(*p)
            assert abs(g - want) <= 1e-14 * max(1.0, abs(want)), (str(e), p, g, want)


@needs_native
def test_constants_tables_and_uses():
    c = {"um": 1.5, "h": "2 * um"}
    e = 4 * constant("um") * y * (constant("h") - y) / constant("h") ** 2
    assert E.evaluate(e, constants=c, y=1.5) == pytest.approx(1.5)
    assert E.uses(e, constants=c) == "y"
    assert E.uses(constant("um") * 2, constants=c) == ""
    ramp = {"ramp": {"x": [0.0, 0.5], "y": [0.0, 1.0]}}
    assert E.evaluate(table("ramp")(t), tables=ramp, t=0.25) == 0.5
    assert E.evaluate(table("ramp")(t), tables=ramp, t=9.0) == 1.0


@needs_native
def test_errors_carry_codes():
    from foehncraft._native import FoehncraftError

    for src, code in [("sin(x", "EXPR001"), ("sinn(x)", "EXPR002"), ("x * t", "EXPR003")]:
        with pytest.raises(FoehncraftError) as ex:
            E.evaluate(src, allowed="xyz")
        assert ex.value.args[0] == code
    assert ne(x, 1) is not None
