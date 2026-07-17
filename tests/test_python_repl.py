from magent.tools.python_repl import PythonREPLTool


def test_prints_are_captured() -> None:
    repl = PythonREPLTool(timeout_seconds=5)
    result = repl.run("print(1 + 1)")
    assert result.ok
    assert result.stdout.strip() == "2"


def test_syntax_error_is_reported_not_raised() -> None:
    repl = PythonREPLTool(timeout_seconds=5)
    result = repl.run("this is not python")
    assert not result.ok
    assert result.error is not None


def test_imports_are_blocked() -> None:
    repl = PythonREPLTool(timeout_seconds=5)
    result = repl.run("import os\nprint(os.getcwd())")
    assert not result.ok
    assert result.error


def test_infinite_loop_times_out() -> None:
    repl = PythonREPLTool(timeout_seconds=1)
    result = repl.run("while True:\n    pass")
    assert result.timed_out
    assert not result.ok
