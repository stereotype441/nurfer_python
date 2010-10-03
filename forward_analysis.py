import re
import unittest
from block import Block
from span import Span
from logical_line import LogicalLine
import tokenization


class SyntacticTypeSpan(Span):
    """Represents a span of characters that all have the same syntactic
    type.  Possible types are:

    normal
    comment
    string
    """
    def __init__(self, full_text, start, end, type):
        Span.__init__(self, full_text, start, end)
        self._type = type

    @property
    def type(self):
        return self._type


def analyze_strings_and_comments(text):
    pos = 0
    while pos < len(text):
        match = tokenization.STRING_OR_COMMENT_START_REGEXP.search(text, pos)
        if match is None:
            yield SyntacticTypeSpan(text, pos, len(text), 'normal')
            return
        if match.start() > pos:
            yield SyntacticTypeSpan(text, pos, match.start(), 'normal')
            pos = match.start()
        if match.lastgroup == 'quote':
            end = tokenization.find_string_end(text, match.end(), match.group())
            type = 'string'
        else:
            assert match.lastgroup == 'comment'
            match = tokenization.EOL_REGEXP.search(text, match.end())
            if match is None:
                end = len(text)
            else:
                end = match.start()
            type = 'comment'
        yield SyntacticTypeSpan(text, pos, end, type)
        pos = end


class TestStringAndCommentAnalysis(unittest.TestCase):
    def test(self):
        test_cases = [('abc # def\n"ghi" """j\nk\nl"""', [('# def', 'comment'), ('"ghi"', 'string'), ('"""j\nk\nl"""', 'string')]),
                      ('"unterminated', [('"unterminated', 'string')]),
                      ('"""unterminated long', [('"""unterminated long', 'string')]),
                      ('"nested \'strings\'"', [("\"nested 'strings'\"", 'string')]),
                      ('# unterminated comment', [('# unterminated comment', 'comment')]),
                      ('abc', []),
                      ('"a#"#b"\n"""c"#""#"""#d', [('"a#"', 'string'), ('#b"', 'comment'), ('"""c"#""#"""', 'string'), ('#d', 'comment')]),
                      ('"line\\\njoining"', [('"line\\\njoining"', 'string')])]
        for text, expected_spans in test_cases:
            spans = [(span.sub_text, span.type) for span in analyze_strings_and_comments(text) if span.type != 'normal']
            self.assertEqual(spans, expected_spans)


class SymbolicNestingSpan(SyntacticTypeSpan):
    """Represents a span of characters that all have the same syntactic
    type and symbolic nesting level.
    """
    def __init__(self, span, symbolic_nesting):
        assert isinstance(span, SyntacticTypeSpan)
        SyntacticTypeSpan.__init__(self, span.full_text, span.start, span.end, span.type)
        self._symbolic_nesting = symbolic_nesting

    @property
    def symbolic_nesting(self):
        return self._symbolic_nesting


def analyze_symbolic_nesting(text):
    symbolic_nesting = 0
    foo = list(analyze_strings_and_comments(text))
    for span in foo:
        if span.type != 'normal':
            yield SymbolicNestingSpan(span, symbolic_nesting)
        else:
            pos = span.start
            while True:
                match = tokenization.NESTING_OPERATOR_REGEXP.search(text, pos, span.end)
                if match is None:
                    break
                elif match.lastgroup == 'open':
                    prev_span, span = span.cut(match.start())
                    yield SymbolicNestingSpan(prev_span, symbolic_nesting)
                    symbolic_nesting += 1
                else:
                    assert match.lastgroup == 'close'
                    prev_span, span = span.cut(match.end())
                    yield SymbolicNestingSpan(prev_span, symbolic_nesting)
                    if symbolic_nesting > 0:
                        symbolic_nesting -= 1
                pos = match.end()
            yield SymbolicNestingSpan(span, symbolic_nesting)


class TestSymbolicNestingAnalysis(unittest.TestCase):
    @staticmethod
    def collect_nested_spans(spans):
        current_text = None
        for span in spans:
            if current_text is not None and current_text_end == span.start and current_nesting == span.symbolic_nesting:
                current_text += span.sub_text
            else:
                if current_text is not None and current_nesting > 0:
                    yield current_text
                current_text = span.sub_text
            current_text_end = span.end
            current_nesting = span.symbolic_nesting
        if current_text is not None and current_nesting > 0:
            yield current_text

    def test(self):
        test_cases = [('', []),
                      ('a', []),
                      ('#', []),
                      ('x(y)z', ['(y)']),
                      ('()()x', ['()', '()']),
                      ('(x', ['(x']),
                      ('(()())()x', ['(()())', '()']),
                      ('((x', ['((x']),
                      ('(")")x', ['(")")']),
                      ('(,),)x', ['(,)']),
                      ('())x()y', ['()', '()']),
                      ('x[y]z', ['[y]']),
                      ('a(b[c)d[e]f', ['(b[c)d[e]f']),
                      ('["]"]x', ['["]"]']),
                      ('x{y}z', ['{y}']),
                      ('a(b{c)d{e}f', ['(b{c)d{e}f']),
                      ('a[b{c]d{e}f', ['[b{c]d{e}f']),
                      ('{"}"}x', ['{"}"}']),
                      ('"abc"  "def"  ("ghi")', ['("ghi")'])]
        for text, expected_spans in test_cases:
            spans = [span_text for span_text in self.collect_nested_spans(analyze_symbolic_nesting(text))]
        self.assertEqual(spans, expected_spans)


def analyze_lines(text):
    """Yield a collection of LogicalLine objects representing all the
    non-blank lines in the document.
    """ 

    def find_unescaped_newline(text, start, end):
        while True:
            pos = text.find('\n', start, end)
            if pos < 1 or text[pos-1] != '\\':
                return pos
            start = pos + 1

    def is_blank_span(span):
        if span.type == 'comment':
            return True
        elif span.type == 'normal':
            return len(span.sub_text.strip()) == 0
        else:
            return False

    def measure_indentation(text):
        indentation = 0
        for chr in text:
            if chr == ' ': indentation += 1
            elif chr == '\t': indentation = (indentation/8)*8 + 8
            else: break
        return indentation

    def analyze_line(spans):
        full_text = spans[0].full_text
        start = spans[0].start
        end = spans[-1].end
        is_blank = all(is_blank_span(span) for span in spans)
        indentation = measure_indentation(full_text[start:end])
        if is_blank:
            return []
        else:
            return [LogicalLine(full_text, start, end, indentation)]

    current_line = []
    for span in analyze_symbolic_nesting(text):
        assert isinstance(span, SymbolicNestingSpan)
        if span.type == 'normal' and span.symbolic_nesting == 0:
            while True:
                newline_pos = find_unescaped_newline(text, span.start, span.end)
                if newline_pos == -1:
                    break
                else:
                    prev_span, span = span.cut(newline_pos + 1)
                    current_line.append(prev_span)
                    if len(current_line) != 0:
                        for logical_line in analyze_line(current_line):
                            yield logical_line
                    current_line = []
        current_line.append(span)
    if len(current_line) != 0:
        for logical_line in analyze_line(current_line):
            yield logical_line


class TestLineAnalysis(unittest.TestCase):
    def test_normal(self):
        text = """def foo():
    "bar"
  # blet
foo(

    )"""
        expected_lines = [('def foo():\n', 0), ('    "bar"\n', 4), ('foo(\n\n    )', 0)]
        computed_lines = [(span.sub_text, span.indentation) for span in analyze_lines(text)]
        self.assertEqual(expected_lines, computed_lines)

    def test_tokens(self):
        test_cases = [("def foo(x=y):", [("def", "def"), ("IDENTIFIER", "foo"), ("(", "("), ("IDENTIFIER", "x"),
                                         ("=", "="), ("IDENTIFIER", "y"), (")", ")"), (":", ":")]),
                      ("x += 'a string' # blah", [("IDENTIFIER", "x"), ("+=", "+="), ("LITERAL", "'a string'")]),
                      ("f(x,#comment\ny)", [("IDENTIFIER", "f"), ("(", "("), ("IDENTIFIER", "x"), (",", ","), ("IDENTIFIER", "y"), (")", ")")])]
        for text, expected_tokens in test_cases:
            actual_tokens = [(token.type, token.sub_text) for token in list(analyze_lines(text))[0].tokens]
            self.assertEqual(actual_tokens, expected_tokens)


class TreeNode(object):
    """An object that can contain child objects."""
    def __init__(self, children):
        self._children = children

    @property
    def children(self):
        return self._children


class Compound(object):
    """Object representing either a simple statement, or a portion of a
    statement followed by a nested suite of statements.

    The following things are compounds:

    (1) x = y

    (2) else:
            print "foo"

    (3) try:
            x = process_file()

    In these examples, the first line is called the "intro" and the
    remaining lines are called the "suite".

    The intro is stored as a LogicalLine object.  The suite is a
    list of Statement objects.
    """

    def __init__(self, intro, suite):
        self._intro = intro
        self._suite = suite

    @property
    def intro(self):
        return self._intro

    @property
    def suite(self):
        return self._suite

    @property
    def end(self):
        if len(self._suite) == 0:
            return self._intro.end
        else:
            return self._suite[-1].end


class Statement(TreeNode):
    """TreeNode representing a complete statement.

    Child objects are the Compound objects that make up the
    statement.  For example, the statement:

    if x:
        y
    else:
        z

    is made up of two compounds:

    if x:
        y

    and

    else:
        z
    """

    @property
    def end(self):
        return self.children[0].end


def analyze_statements(text):
    """Yields Statement objects describing the given text."""

    def extract_compound(lines):
        """Returns a single compound whose first child is the intro part of
        the compound, and whose remaining children (if any),
        constitute the suite.

        Remove those lines from lines.
        """
        # TODO: handle interior colons.
        num_lines_in_compound = 1
        while num_lines_in_compound < len(lines):
            if lines[0].indentation >= lines[num_lines_in_compound].indentation:
                break
            num_lines_in_compound += 1
        compound_intro = lines[0]
        compound_suite = lines[1:num_lines_in_compound]
        del lines[0:num_lines_in_compound]
        analyzed_suite = list(analyze_suite(compound_suite))
        return Compound(compound_intro, analyzed_suite)

    def is_statement_continuation(line):
        return tokenization.STATEMENT_CONTINUATION_REGEXP.match(text, line.start, line.end) is not None

    def analyze_suite(lines):
        """Yields the statements that make up the suite"""
        lines = list(lines)     # So we can modify it safely
        accumulated_compounds = []
        while len(lines) > 0:
            if len(accumulated_compounds) != 0 and not is_statement_continuation(lines[0]):
                yield Statement(accumulated_compounds)
                accumulated_compounds = []
            accumulated_compounds.append(extract_compound(lines))
        if len(accumulated_compounds) != 0:
            yield Statement(accumulated_compounds)

    return analyze_suite(analyze_lines(text))


def illustrate_tree(analysis):
    if isinstance(analysis, list):
        return ','.join(map(illustrate_tree, analysis))
    elif isinstance(analysis, Statement):
        return 'statement(%s)' % illustrate_tree(analysis.children)
    elif isinstance(analysis, Compound):
        contents_illustration = illustrate_tree(analysis.intro)
        if len(analysis.suite) != 0:
            contents_illustration += ':' + illustrate_tree(analysis.suite)
        return 'compound(%s)' % contents_illustration
    elif isinstance(analysis, Block):
        contents_illustration = illustrate_tree(analysis.intro)
        if len(analysis.suite) != 0:
            contents_illustration += ':' + illustrate_tree(analysis.suite)
        return 'block(%s)' % contents_illustration
    else:
        assert isinstance(analysis, Span)
        return "'''%s'''" % analysis.sub_text


class TestStatementAnalysis(unittest.TestCase):
    def test_normal(self):
        text = """def foo(x, y):
    if x == y:
        print "equal"
    else:
        print "not equal"

foo(1,2)
"""
        illustrated_tree = illustrate_tree(list(analyze_statements(text)))
        expected_tree = """statement(compound('''def foo(x, y):
''':statement(compound('''    if x == y:
''':statement(compound('''        print "equal"
'''))),compound('''    else:
''':statement(compound('''        print "not equal"
''')))))),statement(compound('''foo(1,2)
'''))"""
        self.assertEqual(illustrated_tree, expected_tree)


def _flatten_statements_to_blocks(statements):
    """Given an iterable of Statement objects, yield either
    LogicalLine or Block objects representing the separation
    of the statements into blocks.
    """
    for statement in statements:
        for compound in statement.children:
            intro_logical_line = compound.intro
            if tokenization.BLOCK_INTRO_REGEXP.match(intro_logical_line.full_text, intro_logical_line.start, intro_logical_line.end) is not None:
                yield Block(compound.intro, list(_flatten_statements_to_blocks(compound.suite)))
            else:
                yield compound.intro
                for elem in _flatten_statements_to_blocks(compound.suite):
                    yield elem

def flatten_statements_to_block(statements):
    """Given an iterable of Statement objects, return a Block
    representing the toplevel block containing those statements.
    """
    return Block(None, list(_flatten_statements_to_blocks(statements)))


class TestBlockAnalysis(unittest.TestCase):
    def test_normal(self):
        text = """class C(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y
    def print_x(self):
        if self._x is not None:
            print self._x
    print_x_2 = print_x"""
        block = flatten_statements_to_block(analyze_statements(text))
        self.assert_(block.intro is None)
        illustrated_tree = illustrate_tree(block.suite)
        expected_tree = """block('''class C(object):
''':block('''    def __init__(self, x, y):
''':'''        self._x = x
''','''        self._y = y
'''),block('''    def print_x(self):
''':'''        if self._x is not None:
''','''            print self._x
'''),'''    print_x_2 = print_x''')"""
        self.assertEqual(illustrated_tree, expected_tree)


if __name__ == '__main__':
    unittest.main()
