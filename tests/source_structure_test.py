from __future__ import annotations

import ast
from pathlib import Path


class NestedCallableVisitor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.depth = 0
        self.violations: list[str] = []

    def _visit_callable(self, node):
        if self.depth:
            self.violations.append(
                f'{self.path}:{node.lineno}: {type(node).__name__}',
            )
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_FunctionDef = _visit_callable
    visit_AsyncFunctionDef = _visit_callable

    def visit_Lambda(self, node):
        self.violations.append(
            f'{self.path}:{node.lineno}: Lambda',
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if self.depth:
            self.violations.append(
                f'{self.path}:{node.lineno}: ClassDef',
            )
        self.generic_visit(node)


def test_repository_code_has_no_nested_functions_or_lambdas():
    root = Path(__file__).resolve().parents[1]
    paths = list(root.glob('*.py'))
    for directory in ('src', 'examples', 'scripts', 'deploy', 'tests'):
        paths.extend((root / directory).rglob('*.py'))
    violations = []
    for path in sorted(paths):
        visitor = NestedCallableVisitor(path.relative_to(root))
        visitor.visit(ast.parse(path.read_text()))
        violations.extend(visitor.violations)
    assert not violations, '\n'.join(violations)


def test_visitor_detects_nested_callable_kinds():
    sources = (
        'factory = lambda: 1\n',
        'class Owner:\n    factory = lambda: 1\n',
        'def outer():\n    def inner():\n        pass\n',
        'def outer():\n    async def inner():\n        pass\n',
        'def outer():\n    return lambda: 1\n',
        'def outer():\n    class Inner:\n'
        '        def method(self):\n            pass\n',
        'def outer():\n    class Inner:\n        pass\n',
    )
    for source in sources:
        visitor = NestedCallableVisitor(Path('sample.py'))
        visitor.visit(ast.parse(source))
        assert visitor.violations


def test_visitor_accepts_module_helpers_and_class_methods():
    visitor = NestedCallableVisitor(Path('sample.py'))
    visitor.visit(
        ast.parse(
            'def helper():\n    pass\nclass Owner:\n'
            '    def method(self):\n        pass\n',
        ),
    )
    assert not visitor.violations
