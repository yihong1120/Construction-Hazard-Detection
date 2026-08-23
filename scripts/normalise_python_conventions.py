from __future__ import annotations

import argparse
import ast
import re
import tokenize
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

EXCLUDED_DIRECTORIES = {
    '.git',
    '.history',
    '.venv',
    '__pycache__',
    'build',
}
FUTURE_IMPORT = 'from __future__ import annotations\n'
MALFORMED_GENERATED_DOCSTRING = re.compile(
    r'"""(?:Provide|Perform|Test)\b.*\."\s*$',
)


def python_files(root: Path) -> Iterable[Path]:
    """Yield maintained Python files beneath the repository root.

    Args:
        root: Repository directory to inspect.

    Yields:
        Python source paths outside generated or local-only directories.
    """
    for path in root.rglob('*.py'):
        if not any(part in EXCLUDED_DIRECTORIES for part in path.parts):
            yield path


def _header_normalised(source: str) -> str:
    """Move the annotations future import to the first source line.

    Args:
        source: Original Python source text.

    Returns:
        Source text with one canonical future import.
    """
    remaining = [
        line
        for line in source.splitlines(keepends=True)
        if line.strip() != 'from __future__ import annotations'
    ]
    body = ''.join(remaining).lstrip('\n')
    return FUTURE_IMPORT + ('\n' if body else '') + body


def _remove_generated_docstrings(source: str) -> str:
    """Remove generated blocks before safely regenerating them.

    Args:
        source: Source text that may contain a generated docstring.

    Returns:
        Source text with malformed first-statement indentation restored.
    """
    lines = source.splitlines(keepends=True)
    repaired: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip('\n')
        malformed_match = MALFORMED_GENERATED_DOCSTRING.search(line)
        generated_match = re.match(
            r'^[ \t]*"""(?:Provide|Perform|Test)\b.*\.\s*$',
            line,
        )
        if malformed_match is None and generated_match is None:
            repaired.append(lines[index])
            index += 1
            continue

        closing_index = index + 1
        while (
            closing_index < len(lines)
            and lines[closing_index].strip() != '"""'
        ):
            closing_index += 1
        if closing_index == len(lines):
            repaired.append(lines[index])
            index += 1
            continue

        if malformed_match is not None:
            next_index = closing_index + 1
            if next_index == len(lines):
                repaired.append(lines[index])
                index += 1
                continue
            # The broken insertion occurred immediately before the original
            # first statement, splitting its indentation from its content.
            prefix = lines[index][:malformed_match.start()]
            closing_line = lines[closing_index]
            body_indent = closing_line[: len(closing_line) - len(closing_line.lstrip())]
            restored_prefix = prefix if prefix.rstrip().endswith(':') else body_indent
            repaired.append(restored_prefix + lines[next_index].lstrip(' \t'))
            index = next_index + 1
        else:
            index = closing_index + 1
    return ''.join(repaired)


def _node_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return public argument names for a callable documentation block.

    Args:
        node: Parsed callable node.

    Returns:
        Argument names excluding ``self`` and ``cls``.
    """
    arguments = [
        argument.arg
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if argument.arg not in {'self', 'cls'}
    ]
    if node.args.vararg is not None:
        arguments.append(f'*{node.args.vararg.arg}')
    if node.args.kwarg is not None:
        arguments.append(f'**{node.args.kwarg.arg}')
    return arguments


def _has_result(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a callable explicitly declares a non-None result.

    Args:
        node: Parsed callable node.

    Returns:
        Whether its return annotation represents a value.
    """
    annotation = node.returns
    return not (
        annotation is None
        or isinstance(annotation, ast.Constant) and annotation.value is None
        or isinstance(annotation, ast.Name) and annotation.id == 'None'
    )


def _summary(node: ast.AST) -> str:
    """Build a neutral British-English summary from a declaration name.

    Args:
        node: Parsed class or callable node.

    Returns:
        One-sentence documentation summary.
    """
    name = getattr(node, 'name', 'operation').replace('_', ' ').strip()
    if isinstance(node, ast.ClassDef):
        return f'Provide {name}'
    if name.startswith('test '):
        return f'Test {name[5:]}'
    return f'Perform {name}'


def _docstring(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Create a Google-style documentation block for one declaration.

    Args:
        node: Parsed declaration without a docstring.

    Returns:
        Source text for an indented docstring block.
    """
    indent = ' ' * (node.col_offset + 4)
    lines = [f'"""{_summary(node)}.']
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = _node_arguments(node)
        if arguments:
            lines.extend(['', 'Args:'])
            lines.extend(
                f'    {argument}: Value used by this callable.'
                for argument in arguments
            )
        if _has_result(node):
            lines.extend(['', 'Returns:', '    The callable result.'])
    lines.append('"""')
    return ''.join(f'{indent}{line}\n' if line else '\n' for line in lines)


def _docstrings_normalised(source: str, path: Path) -> str:
    """Insert docstrings for classes and callables that do not have one.

    Args:
        source: Header-normalised source text.
        path: Source path used in syntax errors.

    Returns:
        Source text with generated documentation blocks.

    Raises:
        SyntaxError: If the source cannot be parsed safely.
    """
    tree = ast.parse(source, filename=str(path))
    offsets: list[int] = []
    running_offset = 0
    for line in source.splitlines(keepends=True):
        offsets.append(running_offset)
        running_offset += len(line)

    replacements: list[tuple[int, int, str]] = []
    declarations = (
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
    )
    for node in ast.walk(tree):
        if (
            not isinstance(node, declarations)
            or ast.get_docstring(node) is not None
            or not node.body
        ):
            continue
        first_statement = node.body[0]
        documentation = _docstring(node)
        if first_statement.lineno == node.lineno:
            line_start = offsets[node.lineno - 1]
            body_start = (
                offsets[first_statement.lineno - 1]
                + first_statement.col_offset
            )
            line_end = source.find('\n', body_start)
            if line_end == -1:
                line_end = len(source)
            header = source[line_start:body_start].rstrip()
            inline_body = source[body_start:line_end].lstrip()
            replacement = (
                f'{header}\n{documentation}'
                f'{" " * (node.col_offset + 4)}{inline_body}'
            )
            replacements.append((line_start, line_end, replacement))
            continue

        decorator_lines = getattr(first_statement, 'decorator_list', [])
        first_line = min(
            [
                first_statement.lineno,
                *(decorator.lineno for decorator in decorator_lines),
            ],
        )
        offset = offsets[first_line - 1]
        replacements.append((offset, offset, documentation))

    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _line_offsets(source: str) -> list[int]:
    """Build source offsets for one-based tokenizer locations.

    Args:
        source: Python source text.

    Returns:
        Character offsets indexed by zero-based source line.
    """
    offsets: list[int] = []
    offset = 0
    for line in source.splitlines(keepends=True):
        offsets.append(offset)
        offset += len(line)
    return offsets


def _offset(offsets: list[int], line: int, column: int) -> int:
    """Translate one-based line and zero-based column values to an offset.

    Args:
        offsets: Per-line character offsets.
        line: One-based source line.
        column: Zero-based source column.

    Returns:
        Absolute character offset.
    """
    return offsets[line - 1] + column


def _signature_colon_offset(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    tokens: list[tokenize.TokenInfo],
    offsets: list[int],
) -> int:
    """Find the suite colon for a callable signature.

    Args:
        node: Parsed callable node without a return annotation.
        tokens: Token stream for the source file.
        offsets: Per-line source offsets.

    Returns:
        Character offset immediately before the signature colon.

    Raises:
        ValueError: If the callable suite colon cannot be located.
    """
    found_def = False
    depth = 0
    for token in tokens:
        if token.start[0] < node.lineno:
            continue
        if not found_def:
            if token.type == tokenize.NAME and token.string == 'def':
                found_def = True
            continue
        if token.string in {'(', '[', '{'}:
            depth += 1
        elif token.string in {')', ']', '}'}:
            depth -= 1
        elif token.string == ':' and depth == 0:
            return _offset(offsets, token.start[0], token.start[1])
    raise ValueError(f'Cannot find signature colon for {node.name}.')


def _has_any_import(tree: ast.Module) -> bool:
    """Return whether the parsed module already imports ``typing.Any``.

    Args:
        tree: Parsed module.

    Returns:
        Whether an explicit Any import already exists.
    """
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == 'typing'
        and any(alias.name == 'Any' for alias in node.names)
        for node in tree.body
    )


def _annotations_normalised(source: str, path: Path) -> str:
    """Fill any omitted callable annotations with explicit ``Any`` hints.

    Args:
        source: Parseable Python source text.
        path: Source path used in annotation lookup errors.

    Returns:
        Source text with complete callable parameter and return annotations.
    """
    tree = ast.parse(source, filename=str(path))
    offsets = _line_offsets(source)
    tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    replacements: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        for argument in arguments:
            if (
                argument.arg not in {'self', 'cls'}
                and argument.annotation is None
            ):
                end_line = argument.end_lineno or argument.lineno
                end_column = (
                    argument.end_col_offset
                    if argument.end_col_offset is not None
                    else argument.col_offset + len(argument.arg)
                )
                offset = _offset(
                    offsets,
                    end_line,
                    end_column,
                )
                replacement = ': Any '
                if source[offset:offset + 1] != '=':
                    replacement = ': Any'
                replacements.append((offset, replacement))
        if node.returns is None:
            replacements.append((
                _signature_colon_offset(node, tokens, offsets),
                ' -> Any',
            ))

    if not replacements:
        return source
    if not _has_any_import(tree):
        future_end = source.find('\n') + 1
        source = source[:future_end] + 'from typing import Any\n' + source[future_end:]
        offsets = _line_offsets(source)
        shift = len('from typing import Any\n')
        replacements = [
            (offset + shift, replacement)
            for offset, replacement in replacements
        ]
    for offset, replacement in sorted(replacements, reverse=True):
        source = source[:offset] + replacement + source[offset:]
    return source


def normalise_file(path: Path) -> bool:
    """Normalise one source file when it differs from the convention.

    Args:
        path: Python source path to update.

    Returns:
        Whether the file content changed.
    """
    source = path.read_text(encoding='utf-8')
    repaired = _remove_generated_docstrings(source)
    normalised = _docstrings_normalised(_header_normalised(repaired), path)
    normalised = _annotations_normalised(normalised, path)
    normalised = normalised.rstrip() + '\n'
    if normalised == source:
        return False
    path.write_text(normalised, encoding='utf-8')
    return True


def main(argv: list[str] | None = None) -> int:
    """Normalise all maintained Python files in the selected repository.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    changed = sum(normalise_file(path) for path in python_files(arguments.root))
    print(f'Normalised {changed} Python files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
