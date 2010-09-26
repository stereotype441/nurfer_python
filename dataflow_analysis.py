import forward_analysis
import unittest


def do_statements_maybe_read_var(statements, variable):
    # Stupid-simple implementation: just look for the variable name as
    # a token somewhere in one of the statements.
    #
    # TODO: problems with this:
    # - Doesn't see when something is definitely assigned before use
    # - Doesn't see when something is assigned in a generator or lambda (is this a problem?)
    # - No concept of control flow (is this a problem?)
    # - Doesn't know to ignore names defined in nested blocks
    # - Doesn't know what to do about nested functions & classes (actually, is this a problem?)
    for statement in statements:
        for compound in statement.children:
            for token in compound.intro.tokens:
                if token.type == 'IDENTIFIER' and token.sub_text == variable:
                    return True
            if do_statements_maybe_read_var(compound.suite, variable):
                return True
    return False


def do_statements_maybe_write_var(statements, variable):
    # Really stupid-simple implementation: just look for the variable
    # name as a token somewhere in one of the statements.
    #
    # TODO: problems with this:
    # - No concept of control flow (is this a problem?)
    # - Doesn't know how to ignore names defined in nested blocks
    for statement in statements:
        for compound in statement.children:
            for token in compound.intro.tokens:
                if token.type == 'IDENTIFIER' and token.sub_text == variable:
                    return True
            if do_statements_maybe_read_var(compound.suite, variable):
                return True
    return False


class TestDataflowAnalysis(unittest.TestCase):
    def test_do_statements_maybe_read_var(self):
        test_cases = [("x = y", "y", True),
                      ("x = y", "z", False),
                      ("a = b\nc = d", "b", True),
                      ("a = b\nc = d", "d", True),
                      ("a = b\nc = d", "e", False),
                      ("if x:\n    print y\nelse:\n    print z", "x", True),
                      ("if x:\n    print y\nelse:\n    print z", "y", True),
                      ("if x:\n    print y\nelse:\n    print z", "z", True),
                      ("if x:\n    print y\nelse:\n    print z", "w", False)]
        for text, variable, expected_result in test_cases:
            statements = list(forward_analysis.analyze_statements(text))
            result = do_statements_maybe_read_var(statements, variable)
            self.assertEqual(result, expected_result)

    def test_do_statements_maybe_write_var(self):
        test_cases = [("x = y", "x", True),
                      ("x = y", "z", False),
                      ("a = b\nc = d", "a", True),
                      ("a = b\nc = d", "c", True),
                      ("a = b\nc = d", "e", False),
                      ("if len([a for a in b]) != 0:\n    c = d\nelse:\n    e = f", "a", True),
                      ("if len([a for a in b]) != 0:\n    c = d\nelse:\n    e = f", "c", True),
                      ("if len([a for a in b]) != 0:\n    c = d\nelse:\n    e = f", "e", True),
                      ("if len([a for a in b]) != 0:\n    c = d\nelse:\n    e = f", "g", False)]
        for text, variable, expected_result in test_cases:
            statements = list(forward_analysis.analyze_statements(text))
            result = do_statements_maybe_write_var(statements, variable)
            self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
