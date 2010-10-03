import re
import unittest

import forward_analysis
import tokenization


DEF_REGEXP = re.compile(r'\s*def\s+%s\s*\(\s*(?P<args>%s)' % (tokenization.ANONYMOUS_IDENTIFIER_PATTERN, tokenization.TARGET_PATTERN))


def find_bound_names(block):
    bound_names = set()
    if block.intro is not None:
        match = DEF_REGEXP.match(block.intro.full_text, block.intro.start, block.intro.end)
        if match is not None:
            for target in match.group('args').split(','):
                bound_names.add(target.strip())
    for elem in block.suite:
        if isinstance(elem, forward_analysis.LogicalLine):
            bound_names |= set(elem.find_bound_names())
        else:
            assert isinstance(elem, forward_analysis.Block)
            bound_names |= set(elem.intro.find_bound_names())
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
