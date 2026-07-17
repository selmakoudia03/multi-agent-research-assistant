import pytest

from magent.tools.calculator import CalculatorError, CalculatorTool


@pytest.fixture
def calc() -> CalculatorTool:
    return CalculatorTool()


def test_basic_arithmetic(calc: CalculatorTool) -> None:
    assert calc.run("2 + 2") == 4
    assert calc.run("12 * (3 + 4)") == 84
    assert calc.run("10 / 4") == 2.5


def test_allowed_function(calc: CalculatorTool) -> None:
    assert calc.run("sqrt(16)") == 4.0


def test_rejects_disallowed_function(calc: CalculatorTool) -> None:
    with pytest.raises(CalculatorError):
        calc.run("__import__('os').system('echo pwned')")


def test_rejects_name_access(calc: CalculatorTool) -> None:
    with pytest.raises(CalculatorError):
        calc.run("open('/etc/passwd').read()")


def test_rejects_invalid_syntax(calc: CalculatorTool) -> None:
    with pytest.raises(CalculatorError):
        calc.run("2 +")
