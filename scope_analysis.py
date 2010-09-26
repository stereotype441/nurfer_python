import re
import unittest

import forward_analysis


ANONYMOUS_IDENTIFIER_PATTERN = r'[a-zA-Z_]\w*'
TARGET_PATTERN = r'%s(\s*,\s*%s)*' % (ANONYMOUS_IDENTIFIER_PATTERN, ANONYMOUS_IDENTIFIER_PATTERN)
DEF_REGEXP = re.compile(r'\s*def\s+%s\s*\(\s*(?P<args>%s)' % (ANONYMOUS_IDENTIFIER_PATTERN, TARGET_PATTERN))
IMPORT_PATTERN = (r'(from\s+%s(\.%s)*\s+)?import\s+(?P<import>%s)'
                  % (ANONYMOUS_IDENTIFIER_PATTERN, ANONYMOUS_IDENTIFIER_PATTERN, TARGET_PATTERN))
BOUND_NAME_REGEXP = re.compile(r'\s*(?:(?P<assignment>%s)\s*=|for\s+(?P<for>%s)|%s|(class|def)\s+(?P<class_or_def>%s))' %
                               (TARGET_PATTERN, TARGET_PATTERN, IMPORT_PATTERN, ANONYMOUS_IDENTIFIER_PATTERN))


def find_bound_names_on_line(line):
    match = BOUND_NAME_REGEXP.match(line.full_text, line.start, line.end)
    if match is not None:
        for target in match.group(match.lastgroup).split(','):
            yield target.strip()


def find_bound_names(block):
    bound_names = set()
    if block.intro is not None:
        match = DEF_REGEXP.match(block.intro.full_text, block.intro.start, block.intro.end)
        if match is not None:
            for target in match.group('args').split(','):
                bound_names.add(target.strip())
    for elem in block.suite:
        if isinstance(elem, forward_analysis.LogicalLine):
            bound_names |= set(find_bound_names_on_line(elem))
        else:
            assert isinstance(elem, forward_analysis.Block)
            bound_names |= set(find_bound_names_on_line(elem.intro))
    return bound_names


class TestFindBoundNames(unittest.TestCase):
    def test_find_bound_names(self):
        test_cases = [('x = y', ['x']),
                      ('def f(x, y)\n    return y, z', ['f']),
                      ('z = "foo"\ndef f(x, y)\n    return y, z', ['z', 'f']),
                      ('def f(x, y)\n    return y.z', ['f']),
                      ('def f(x, y)\n    z = x\n    return (y, z)', ['f']),
                      ('def f():\n    print "f"\nf()', ['f']),
                      ('class MyClass(object):\n    pass\nprint MyClass()', ['MyClass']),
                      ('class MyClass(foo):\n    pass\nprint MyClass()', ['MyClass'])]


if __name__ == '__main__':
    unittest.main()
