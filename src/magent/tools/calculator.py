"""A calculator tool that never calls `eval`.

Arithmetic expressions are parsed into an AST and walked manually, so only
numeric literals, `+ - * / // % **`, unary +/-, parentheses and a small
allowlist of `math` functions are reachable. This is the kind of tool an
LLM agent will call with untrusted, model-generated input, so it must be
safe by construction rather than by blocklist.
"""

from __future__ import annotations

import ast
import math
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
}


class CalculatorError(ValueError):
    pass


class CalculatorTool:
    """Evaluate a single arithmetic expression string safely."""

    name = "calculator"

    def run(self, expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise CalculatorError(f"Invalid expression: {expression!r}") from exc
        return self._eval(tree.body)

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", None)
            if func_name not in _ALLOWED_FUNCS:
                raise CalculatorError(f"Function not allowed: {func_name!r}")
            args = [self._eval(arg) for arg in node.args]
            return _ALLOWED_FUNCS[func_name](*args)
        raise CalculatorError(f"Disallowed expression element: {ast.dump(node)}")
