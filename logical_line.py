from span import Span
import tokenization


KEYWORDS = ['and', 'as', 'assert', 'break', 'class', 'continue',
            'def', 'del', 'elif', 'else', 'except', 'exec', 'finally',
            'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return',
            'try', 'while', 'with', 'yield']


class LogicalLine(Span):
    """Span representing a single logical line in the document."""

    def __init__(self, full_text, start, end, indentation):
        Span.__init__(self, full_text, start, end)
        self._indentation = indentation
        self._tokens = list(self._tokenize())

    def _tokenize(self):
        pos = self.start
        while pos <= self.end:
            match = tokenization.TOKEN_REGEXP.search(self.full_text, pos, self.end)
            if match is None:
                break
            start, end = match.span()
            if match.lastgroup == 'identifier':
                if match.group() in KEYWORDS:
                    type = match.group()
                else:
                    type = 'IDENTIFIER'
            elif match.lastgroup == 'symbol':
                type = match.group()
            elif match.lastgroup == 'quote':
                type = 'LITERAL'
                quote_type = match.group('quote')
                end = tokenization.find_string_end(self.full_text, end, quote_type)
            elif match.lastgroup == 'comment':
                pos = end
                continue
            yield tokenization.Token(self.full_text, type, start, end)
            pos = end

    @property
    def indentation(self):
        return self._indentation

    @property
    def tokens(self):
        return self._tokens
