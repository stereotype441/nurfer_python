from logical_line import LogicalLine
import tokenization
import re
import unittest


DEF_REGEXP = re.compile(r'\s*def\s+%s\s*\(\s*(?P<args>%s)' % (tokenization.ANONYMOUS_IDENTIFIER_PATTERN, tokenization.TARGET_PATTERN))


class Block(object):
    """Represents a block.  That is, either a function or a class.

    The first line of the definition is called the "intro" and the
    remaining lines are called the "suite".

    The intro is stored as a LogicalLine object.  The suite is a
    list of Block or LogicalLine objects.

    For the toplevel block, the intro is None.
    """
    # TODO: Handle decorators.

    def __init__(self, intro, suite):
        self._intro = intro
        self._suite = suite

    @property
    def intro(self):
        return self._intro

    @property
    def suite(self):
        return self._suite

    def find_bound_names(self):
        bound_names = set()
        if self.intro is not None:
            match = DEF_REGEXP.match(self.intro.full_text, self.intro.start, self.intro.end)
            if match is not None:
                for target in match.group('args').split(','):
                    bound_names.add(target.strip())
        for elem in self.suite:
            if isinstance(elem, LogicalLine):
                bound_names |= set(elem.find_bound_names())
            else:
                assert isinstance(elem, Block)
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
        # TODO: write the test


if __name__ == '__main__':
    unittest.main()
