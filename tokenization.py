from span import Span
import re


SHORT_STRING_CONTENTS_REGEXP = re.compile('\\\\.|(?P<quote>\'|")|(?P<eol>$)', re.DOTALL | re.MULTILINE)
LONG_STRING_CONTENTS_REGEXP = re.compile('\\\\.|(?P<quote>\'{3}|"{3})', re.DOTALL)
SIMPLE_STRING_START_PATTERN = '(?P<quote>\'(\'\')?|"("")?)' # Ignores leading 'u'/'r'
ANONYMOUS_IDENTIFIER_PATTERN = r'[a-zA-Z_]\w*'
IDENTIFIER_PATTERN = r'(?P<identifier>%s)' % ANONYMOUS_IDENTIFIER_PATTERN
COMMENT_PATTERN = r'(?P<comment>#.*$)'
SYMBOL_PATTERN = r'(?P<symbol>!=|%=|&=|\*\*|\*\*=|\*=|\+=|-=|//|//=|/=|<<|<<=|<=|<>|==|>=|>>|>>=|\^=|\|=|[!%&()*+,\-./:;<=>@[\]^`{|}~])'
TOKEN_REGEXP = re.compile('%s|%s|%s|%s' % (SIMPLE_STRING_START_PATTERN, IDENTIFIER_PATTERN, COMMENT_PATTERN, SYMBOL_PATTERN), re.MULTILINE)
COMMENT_START_PATTERN = r'(?P<comment>#)'
STRING_OR_COMMENT_START_REGEXP = re.compile('%s|%s' % (SIMPLE_STRING_START_PATTERN, COMMENT_START_PATTERN))
EOL_REGEXP = re.compile('$', re.MULTILINE)
NESTING_OPERATOR_REGEXP = re.compile(r'(?P<open>[[({])|(?P<close>[])}])')
STATEMENT_CONTINUATION_REGEXP = re.compile(r'\s*(elif|else|except|finally)\b')
BLOCK_INTRO_REGEXP = re.compile(r'\s*(def|class)\b')
ANONYMOUS_IDENTIFIER_PATTERN = r'[a-zA-Z_]\w*'
TARGET_PATTERN = r'%s(\s*,\s*%s)*' % (ANONYMOUS_IDENTIFIER_PATTERN, ANONYMOUS_IDENTIFIER_PATTERN)


class Token(Span):
    """Stores the location of a token in a document, and its type and text.

    Possible types are:
      'IDENTIFIER'
      'LITERAL' (e.g. string)
      any keyword (lowercase)
      any symbol
    """

    def __init__(self, full_text, token_type, start, end):
        Span.__init__(self, full_text, start, end)
        self._token_type = token_type

    @property
    def type(self):
        return self._token_type

    def __str__(self):
        return self.sub_text


def find_string_end(text, pos, quote_type):
    if len(quote_type) == 1:
        re = SHORT_STRING_CONTENTS_REGEXP
    else:
        re = LONG_STRING_CONTENTS_REGEXP
    while pos <= len(text):
        match = re.search(text, pos)
        if match is None:
            break
        if match.lastgroup == 'quote' and match.group() == quote_type:
            return match.end()
        elif match.lastgroup == 'eol':
            return match.start()
        pos = match.end()
    return len(text) # Unterminated string.


