import optparse
import re
import unittest
import sys

import forward_analysis
import scope_analysis


BUILTIN_NAMES = ['ArithmeticError', 'AssertionError',
                 'AttributeError', 'BaseException',
                 'DeprecationWarning', 'EOFError', 'Ellipsis',
                 'EnvironmentError', 'Exception', 'False',
                 'FloatingPointError', 'FutureWarning',
                 'GeneratorExit', 'IOError', 'ImportError',
                 'ImportWarning', 'IndexError', 'KeyError',
                 'KeyboardInterrupt', 'LookupError', 'MemoryError',
                 'NameError', 'None', 'NotImplemented',
                 'NotImplementedError', 'OSError', 'OverflowError',
                 'PendingDeprecationWarning', 'ReferenceError',
                 'RuntimeError', 'RuntimeWarning', 'StandardError',
                 'StopIteration', 'SyntaxError', 'SyntaxWarning',
                 'SystemError', 'SystemExit', 'True', 'TypeError',
                 'UnboundLocalError', 'UnicodeDecodeError',
                 'UnicodeEncodeError', 'UnicodeError',
                 'UnicodeTranslateError', 'UnicodeWarning',
                 'UserWarning', 'ValueError', 'Warning',
                 'WindowsError', 'ZeroDivisionError',
                 '__builtins__', '__doc__', '__import__',
                 '__name__', 'abs', 'all', 'any', 'apply',
                 'basestring', 'bool', 'buffer', 'callable', 'chr',
                 'classmethod', 'cmp', 'coerce', 'compile',
                 'complex', 'delattr', 'dict', 'dir', 'divmod',
                 'enumerate', 'eval', 'execfile', 'file', 'filter',
                 'float', 'frozenset', 'getattr', 'globals',
                 'hasattr', 'hash', 'help', 'hex', 'id', 'input',
                 'int', 'intern', 'isinstance', 'issubclass',
                 'iter', 'len', 'list', 'locals', 'long', 'map',
                 'max', 'min', 'object', 'oct', 'open', 'ord',
                 'pow', 'property', 'range', 'raw_input', 'reduce',
                 'reload', 'repr', 'reversed', 'round', 'set',
                 'setattr', 'slice', 'sorted', 'staticmethod',
                 'str', 'sum', 'super', 'tuple', 'type', 'unichr',
                 'unicode', 'vars', 'xrange', 'zip']
IMPORT_REGEXP = re.compile(r'\s*(import|from)\b')
DOUBLE_QUOTE_REGEXP = re.compile('"')


def find_name_references_on_line(line):
    token_num = 0
    while token_num < len(line.tokens):
        if line.tokens[token_num].type == 'IDENTIFIER':
            yield line.tokens[token_num]
        elif line.tokens[token_num].type == '.':
            token_num += 1      # skip the next IDENTIFIER.
        token_num += 1


class SimpleDiff(object):
    def __init__(self, diff_start, diff_end, new_text):
        self._diff_start = diff_start
        self._diff_end = diff_end
        self._new_text = new_text

    @property
    def diff_start(self):
        return self._diff_start

    @property
    def diff_end(self):
        return self._diff_end

    @property
    def new_text(self):
        return self._new_text


def elisp_escape_string(str):
    return '"%s"' % DOUBLE_QUOTE_REGEXP.sub(r'\"', str)


class QuickFix(object):
    def __init__(self, text, diff_fn):
        self._text = text
        self._diff_fn = diff_fn

    @property
    def text(self):
        return self._text

    @property
    def diff_fn(self):
        return self._diff_fn

    @property
    def elisp_string(self):
        diff = self._diff_fn()
        return '(%s %d %d %s)' % (elisp_escape_string(self._text), diff.diff_start, diff.diff_end, elisp_escape_string(diff.new_text))


def is_import_statement(statement):
    if len(statement.children) == 0:
        assert False
        return False
    intro_logical_line = statement.children[0].intro
    return IMPORT_REGEXP.match(intro_logical_line.full_text, intro_logical_line.start, intro_logical_line.end) is not None


def fixup_add_import(toplevel_statements, import_name):
    add_import_loc = 0
    for statement in toplevel_statements:
        if is_import_statement(statement):
            add_import_loc = statement.end
        else:
            break
    return SimpleDiff(add_import_loc, add_import_loc, 'import %s\n' % import_name)


def find_unbound_tokens(toplevel_statements, block, names_bound_in_enclosing_scopes):
    assert isinstance(block, forward_analysis.Block)
    bound_names = names_bound_in_enclosing_scopes | scope_analysis.find_bound_names(block)
    if block.intro is None:
        extended_suite = block.suite
    else:
        extended_suite = [block.intro] + block.suite
    for elem in extended_suite:
        if isinstance(elem, forward_analysis.LogicalLine):
            for token in find_name_references_on_line(elem):
                if token.sub_text not in bound_names:
                    fixup = QuickFix('import %s' % token.sub_text, lambda: fixup_add_import(toplevel_statements, token.sub_text))
                    yield token, fixup
        else:
            assert isinstance(elem, forward_analysis.Block)
            for token, fixup in find_unbound_tokens(toplevel_statements, elem, bound_names):
                # TODO: names bound in classes don't carry downward.
                yield token, fixup


def find_unbound_tokens_in_text(text):
    toplevel_statements = list(forward_analysis.analyze_statements(text))
    block = forward_analysis.flatten_statements_to_block(toplevel_statements)
    return find_unbound_tokens(toplevel_statements, block, set(BUILTIN_NAMES))


class TestFindUnboundTokens(unittest.TestCase):
    def test_unbound(self):
        test_cases = [('x = y', [('y', 0)]),
                      ('def f(x, y)\n    return y, z', [('z', 1)]),
                      ('z = "foo"\ndef f(x, y)\n    return y, z', []),
                      ('def f(x, y)\n    return y.z', []),
                      ('def f(x, y)\n    z = x\n    return (y, z)', []),
                      ('def f():\n    print "f"\nf()', []),
                      ('class MyClass(object):\n    pass\nprint MyClass()', []),
                      ('class MyClass(foo):\n    pass\nprint MyClass()', [('foo', 0)])]
        for text, expected in test_cases:
            unbound_tokens = find_unbound_tokens_in_text(text)
            tokens_and_line_numbers = [(token.sub_text, token.full_text.count('\n', 0, token.start)) for token, fixup in unbound_tokens]
            self.assertEqual(tokens_and_line_numbers, expected)

    def test_fixups(self):
        text = """import foo
import bar
from bar import baz

def f(x):
    sys.exit(1)"""
        unbound_tokens = list(find_unbound_tokens_in_text(text))
        self.assertEqual(1, len(unbound_tokens))
        token, fixup = unbound_tokens[0]
        self.assertEqual('import sys', fixup.text)
        diff = fixup.diff_fn()
        self.assertEqual(text.find('\ndef'), diff.diff_start)
        self.assertEqual(diff.diff_start, diff.diff_end)
        self.assertEqual(diff.new_text, 'import sys\n')


if __name__ == '__main__':
    # Unit test if no command line options given
    if len(sys.argv) == 1:
        unittest.main()
        sys.exit(0)

    # Interpret command line
    parser = optparse.OptionParser()
    parser.add_option('-s', '--from-stdin', action='store_true',
                      dest='input_from_stdin', help='Take input from stdin instead of a file')
    parser.add_option('-f', '--show-fixups', action='store_true', dest='show_fixups',
                      help='Show full fixups')
    parser.add_option('-n', '--only-item', action='store', type='int', dest='only_item', metavar='n', default=None,
                      help='Only show item n (0-based)')
    options, args = parser.parse_args()

    # Read the file
    if options.input_from_stdin:
        text = sys.stdin.read()
    else:
        filename = args[0]
        file = open(filename, 'r')
        text = file.read()
        file.close()

    # Print all names that aren't bound.
    unbound_tokens = find_unbound_tokens_in_text(text)
    print '('
    for n, (token, fixup) in enumerate(unbound_tokens):
        if options.only_item is None or options.only_item == n:
            if options.show_fixups:
                fixup_string = fixup.elisp_string
            else:
                fixup_string = elisp_escape_string(fixup.text)
            print '(%d %d %s)' % (token.start, token.end, fixup_string)
    print ')'
