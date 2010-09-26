import re
import unittest


WORD_REGEXP = re.compile(r'[a-zA-Z0-9_]+')
KEYWORDS = ['and', 'as', 'assert', 'break', 'class', 'continue',
            'def', 'del', 'elif', 'else', 'except', 'exec', 'finally',
            'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return',
            'try', 'while', 'with', 'yield']


class Token(object):
    def __init__(self, type, str):
        self._type = type
        self._str = str

    @property
    def type(self):
        return self._type

    @property
    def str(self):
        return self._str


def word_type_f(s):
    if s in KEYWORDS:
        return s
    else:
        return 'identifier'


def make_string_pattern(q):
    ending_pattern = r'\Z'
    if len(q) == 1:
        ending_pattern += '|(?=[\r\n])'
    return q + r'([^\\]|\\.)*?(' + q + '|' + ending_pattern + ')'


STRING_PATTERN = '[uU]?[rR]?(%s)' % '|'.join([make_string_pattern(q) for q in ["'''", '"""', "'", '"']])
TOKENIZERS = [(re.compile('\r\n?|\n'), lambda s: 'newline'),
              (re.compile('#[^\r\n]*(?=[\r\n]|\Z)'), lambda s: 'comment'),
              (re.compile(STRING_PATTERN), lambda s: 'string'),
              (re.compile('[a-zA-Z_][a-zA-Z0-9_]*'), word_type_f),
              (re.compile('[0-9]+'), lambda s: 'number'),
              (re.compile('<='), lambda s: s),
              (re.compile('[ \t\f\v]+'), lambda s: 'space'),
              (re.compile('.'), lambda s: s)]


def untested(x):
    raise Exception('untested')


def pre_tokenize(str):
    """Yields a sequence of tokens of the type Token.

    Notes:
    - Comments are returned as tokens of type 'comment', and line
      continuations are returned as tokens of type
      'line-continuation'.
    - Newlines are returned as tokens of type 'newline'.  Implicit
      line joining is not performed.
    - Contiguous ranges of whitespace are returned as tokens of type
      'space'.
    - Encoding declarations are not processed.
    """
    i = 0
    while i < len(str):
        for regexp, type_f in TOKENIZERS:
            match = regexp.match(str, i)
            if match is not None:
                yield Token(type_f(match.group()), match.group())
                i = match.end()
                break
        else:
            # No match - this should never happen.
            assert False
            i = i + 1


def all_combinations(arr, n):
    if n <= 0:
        return [[]]
    else:
        result = []
        for a in all_combinations(arr, n - 1):
            for b in arr:
                result.append(a + [b])
    return result


class TestPreTokenize(unittest.TestCase):
    def check_tokenization(self, text, expected_tokens):
        tokens = list(pre_tokenize(text))
        tokens = [(token.type, token.str) for token in tokens]
        self.assertEqual(tokens, expected_tokens, '%r: %r != %r' % (text, tokens, expected_tokens))

    def test(self):
        test_cases = [('', []),
                      ('\r', [('newline', '\r')]),
                      (' \n', [('space', ' '), ('newline', '\n')]),
                      ('\r\n', [('newline', '\r\n')]),
                      ('a\n\fb', [('identifier', 'a'), ('newline', '\n'), ('space', '\f'), ('identifier', 'b')]),
                      ('# a comment', [('comment', '# a comment')]),
                      ('# a comment\n', [('comment', '# a comment'), ('newline', '\n')]),
                      ('# a comment\r\n', [('comment', '# a comment'), ('newline', '\r\n')]),
                      ('# a comment\r', [('comment', '# a comment'), ('newline', '\r')]),
                      ('a \\', [('identifier', 'a'), ('space', ' '), ('\\', '\\')]),
                      ('a \\\rb', [('identifier', 'a'), ('space', ' '), ('\\', '\\'), ('newline', '\r'), ('identifier', 'b')]),
                      ('a b\tc\fd', [('identifier', 'a'), ('space', ' '), ('identifier', 'b'), ('space', '\t'), ('identifier', 'c'), ('space', '\f'), ('identifier', 'd')]),
                      ('ab', [('identifier', 'ab')]),
                      ('(\n)\n', [('(', '('), ('newline', '\n'), (')', ')'), ('newline', '\n')])]
        for text, expected_tokens in test_cases:
            self.check_tokenization(text, expected_tokens)

    def test_strings(self):
        complete_strings = []
        quote_types = ["'", '"', "'''", '"""']
        for string_prefix in ['', 'r', 'u', 'ur', 'R', 'U', 'UR', 'Ur', 'uR']:
            for string_type in quote_types:
                possible_contents = ['a', r'\"', r"\'"]
                possible_contents.extend([q for q in quote_types if q.find(string_type) == -1])
                if len(string_type) == 3:
                    possible_contents.extend(['\r', '\n'])
                for num_parts in xrange(4):
                    for contents_parts in all_combinations(possible_contents, num_parts):
                        if len(contents_parts) > 0 and len(string_type) == 3 and contents_parts[-1] == string_type[0]:
                            # This would be a string like '''a'''', which we don't expect to parse as usual.
                            continue
                        contents = ''.join(contents_parts)
                        str = string_prefix + string_type + contents + string_type
                        self.check_tokenization(str, [('string', str)])
                        self.check_tokenization(str + 'a', [('string', str), ('identifier', 'a')])
                        incomplete_str = string_prefix + string_type + contents
                        self.check_tokenization(incomplete_str, [('string', incomplete_str)])
                        if len(string_type) == 1:
                            self.check_tokenization(incomplete_str + '\r', [('string', incomplete_str), ('newline', '\r')])
                            self.check_tokenization(incomplete_str + '\n', [('string', incomplete_str), ('newline', '\n')])


def compute_indentation(str):
    level = 0
    for c in str:
        if c == ' ':
            level += 1
        elif c == '\t':
            level = ((level + 8)//8)*8
    return level


class TestComputeIndentation(unittest.TestCase):
    def test(self):
        test_cases = [('', 0), (' ', 1), ('    ', 4), ('\t', 8), ('  \t', 8), ('       \t', 8), ('        \t', 16), (' \t ', 9), ('\f ', 1)]
        for text, expected_indentation in test_cases:
            indentation = compute_indentation(text)
            self.assertEqual(indentation, expected_indentation, '%r: %r != %r' % (text, indentation, expected_indentation))


def tokenize(str):
    """Yields a sequence of tokens of the form (type, str), where type is
    the type of token and str is the contents of the token.

    This function performs implicit line joining and indentation
    analysis.  It discards comments, line continuations, and blank
    lines.  It does not process encoding declarations.  It does not
    detect indentation errors.
    """
    indent_stack = [0]
    parenthetical_nesting_level = 0
    backslash_token = None
    between_lines = True
    indentation_level = 0
    for token in pre_tokenize(str):
        type = token.type
        # Update parenthetical nesting level so we'll know whether to
        # implicitly join lines
        if type in ['[', '(', '{']:
            parenthetical_nesting_level += 1
        elif type in [']', ')', '}'] and parenthetical_nesting_level > 0:
            parenthetical_nesting_level -= 1
        # If the previous token was a backslash, then absorb a newline
        # (asuming there is one).  Otherwise pretend the backslash was
        # a normal token.
        if backslash_token is not None:
            assert not between_lines
            if type == 'newline':
                backslash_token = None
                continue
            yield backslash_token
            backslash_token = None
        # If we're between lines, handle indentation and line start.
        if between_lines:
            if type == 'space':
                indentation_level = compute_indentation(token.str)
            elif type in ['newline', 'comment']:
                pass
            else:
                # A line is starting
                while indentation_level < indent_stack[-1]:
                    yield Token('dedent', '')
                    del indent_stack[-1]
                if indentation_level > indent_stack[-1]:
                    yield Token('indent', '')
                    indent_stack.append(indentation_level)
                between_lines = False
        # If this token is a backslash, store it away for possible
        # absorption of a newline.
        if type == '\\':
            backslash_token = token
            continue
        # If it's a newline, then absorb it if we're joining lines.
        # Also reset the indentation.
        elif type == 'newline':
            indentation_level = 0
            if between_lines or parenthetical_nesting_level > 0:
                continue
            between_lines = True
        # If it's a comment or space, then don't pass it to the
        # output.
        elif type in ['comment', 'space']:
            continue
        # Otherwise, output the token.
        yield token
    # At the end, output any trailing backslash, and dedent
    # everything.
    if backslash_token is not None:
        yield backslash_token
    for i in xrange(len(indent_stack) - 1):
        yield Token('dedent', '')


class TestTokenize(unittest.TestCase):
    def test(self):
        test_cases = [('(\n)\n', [('(', '('), (')', ')'), ('newline', '\n')]),
                      ('[\n]\n', [('[', '['), (']', ']'), ('newline', '\n')]),
                      ('{\n}\n', [('{', '{'), ('}', '}'), ('newline', '\n')]),
                      ('{(\n)\n}\n', [('{', '{'), ('(', '('), (')', ')'), ('}', '}'), ('newline', '\n')]),
                      ('# comment\nfoo\\\nbar', [('identifier', 'foo'), ('identifier', 'bar')]),
                      ('# comment\nfoo\n\n# comment\n\nbar', [('identifier', 'foo'), ('newline', '\n'), ('identifier', 'bar')]),
                      ('# comment\nfoo\n\n  \\\n  \n\nbar', [('identifier', 'foo'), ('newline', '\n'), ('indent', ''), ('newline', '\n'), ('dedent', ''), ('identifier', 'bar')]),
                      ('a\\\nb', [('identifier', 'a'), ('identifier', 'b')]),
                      ("month_names = ['Januari', 'Februari', 'Maart',      # These are the\n               'April',   'Mei',      'Juni',       # Dutch names\n               'Juli',    'Augustus', 'September',  # for the months\n               'Oktober', 'November', 'December']   # of the year\n",
                       [('identifier', "month_names"),
                        ('=', "="),
                        ('[', "["),
                        ('string', "'Januari'"),
                        (',', ","),
                        ('string', "'Februari'"),
                        (',', ","),
                        ('string', "'Maart'"),
                        (',', ","),
                        ('string', "'April'"),
                        (',', ","),
                        ('string', "'Mei'"),
                        (',', ","),
                        ('string', "'Juni'"),
                        (',', ","),
                        ('string', "'Juli'"),
                        (',', ","),
                        ('string', "'Augustus'"),
                        (',', ","),
                        ('string', "'September'"),
                        (',', ","),
                        ('string', "'Oktober'"),
                        (',', ","),
                        ('string', "'November'"),
                        (',', ","),
                        ('string', "'December'"),
                        (']', "]"),
                        ('newline', '\n')]),
                      ('def perm(l):\n        # Compute the list of all permutations of l\n    if len(l) <= 1:\n                  return [l]\n    r = []\n    for i in range(len(l)):\n             s = l[:i] + l[i+1:]\n             p = perm(s)\n             for x in p:\n              r.append(l[i:i+1] + x)\n    return r\n',
                       [('def', 'def'),
                        ('identifier', 'perm'),
                        ('(', '('),
                        ('identifier', 'l'),
                        (')', ')'),
                        (':', ':'),
                        ('newline', '\n'),
                        ('indent', ''),
                        ('if', 'if'),
                        ('identifier', 'len'),
                        ('(', '('),
                        ('identifier', 'l'),
                        (')', ')'),
                        ('<=', '<='),
                        ('number', '1'),
                        (':', ':'),
                        ('newline', '\n'),
                        ('indent', ''),
                        ('return', 'return'),
                        ('[', '['),
                        ('identifier', 'l'),
                        (']', ']'),
                        ('newline', '\n'),
                        ('dedent', ''),
                        ('identifier', 'r'),
                        ('=', '='),
                        ('[', '['),
                        (']', ']'),
                        ('newline', '\n'),
                        ('for', 'for'),
                        ('identifier', 'i'),
                        ('in', 'in'),
                        ('identifier', 'range'),
                        ('(', '('),
                        ('identifier', 'len'),
                        ('(', '('),
                        ('identifier', 'l'),
                        (')', ')'),
                        (')', ')'),
                        (':', ':'),
                        ('newline', '\n'),
                        ('indent', ''),
                        ('identifier', 's'),
                        ('=', '='),
                        ('identifier', 'l'),
                        ('[', '['),
                        (':', ':'),
                        ('identifier', 'i'),
                        (']', ']'),
                        ('+', '+'),
                        ('identifier', 'l'),
                        ('[', '['),
                        ('identifier', 'i'),
                        ('+', '+'),
                        ('number', '1'),
                        (':', ':'),
                        (']', ']'),
                        ('newline', '\n'),
                        ('identifier', 'p'),
                        ('=', '='),
                        ('identifier', 'perm'),
                        ('(', '('),
                        ('identifier', 's'),
                        (')', ')'),
                        ('newline', '\n'),
                        ('for', 'for'),
                        ('identifier', 'x'),
                        ('in', 'in'),
                        ('identifier', 'p'),
                        (':', ':'),
                        ('newline', '\n'),
                        ('indent', ''),
                        ('identifier', 'r'),
                        ('.', '.'),
                        ('identifier', 'append'),
                        ('(', '('),
                        ('identifier', 'l'),
                        ('[', '['),
                        ('identifier', 'i'),
                        (':', ':'),
                        ('identifier', 'i'),
                        ('+', '+'),
                        ('number', '1'),
                        (']', ']'),
                        ('+', '+'),
                        ('identifier', 'x'),
                        (')', ')'),
                        ('newline', '\n'),
                        ('dedent', ''),
                        ('dedent', ''),
                        ('return', 'return'),
                        ('identifier', 'r'),
                        ('newline', '\n'),
                        ('dedent', '')]
                       ),
                      ('a\n\fb', [('identifier', 'a'), ('newline', '\n'), ('identifier', 'b')]),
                      ('if 1900 < year < 2100 and 1 <= month <= 12 \\\n   and 1 <= day <= 31 and 0 <= hour < 24 \\\n   and 0 <= minute < 60 and 0 <= second < 60:   # Looks like a valid date\n        return 1',
                       [('if', 'if'),
                        ('number', '1900'),
                        ('<', '<'),
                        ('identifier', 'year'),
                        ('<', '<'),
                        ('number', '2100'),
                        ('and', 'and'),
                        ('number', '1'),
                        ('<=', '<='),
                        ('identifier', 'month'),
                        ('<=', '<='),
                        ('number', '12'),
                        ('and', 'and'),
                        ('number', '1'),
                        ('<=', '<='),
                        ('identifier', 'day'),
                        ('<=', '<='),
                        ('number', '31'),
                        ('and', 'and'),
                        ('number', '0'),
                        ('<=', '<='),
                        ('identifier', 'hour'),
                        ('<', '<'),
                        ('number', '24'),
                        ('and', 'and'),
                        ('number', '0'),
                        ('<=', '<='),
                        ('identifier', 'minute'),
                        ('<', '<'),
                        ('number', '60'),
                        ('and', 'and'),
                        ('number', '0'),
                        ('<=', '<='),
                        ('identifier', 'second'),
                        ('<', '<'),
                        ('number', '60'),
                        (':', ':'),
                        ('newline', '\n'),
                        ('indent', ''),
                        ('return', 'return'),
                        ('number', '1'),
                        ('dedent', '')]),
                      ('a \\', [('identifier', 'a'), ('\\', '\\')]),
                      ('a \\\rb', [('identifier', 'a'), ('identifier', 'b')]),
                      ('a \\\nb', [('identifier', 'a'), ('identifier', 'b')]),
                      ('a \\\r\nb', [('identifier', 'a'), ('identifier', 'b')])]
        for text, expected_tokens in test_cases:
            tokens = list(tokenize(text))
            tokens = [(token.type, token.str) for token in tokens]
            self.assertEqual(tokens, expected_tokens, '%r: %r != %r' % (text, tokens, expected_tokens))


if __name__ == '__main__':
    unittest.main()
