import tokenize


class TreeNode(object):
    def __init__(self, type, *child_nodes):
        self._type = type
        self._child_nodes = child_nodes

    @property
    def type(self):
        return self._type

    @property
    def child_nodes(self):
        return self._child_nodes

    def append(self, *new_child_nodes):
        self._child_nodes = self._child_nodes + new_child_nodes

    def de_append(self):
        result = self._child_nodes[-1]
        self._child_nodes = self._child_nodes[0:-1]
        return result


def parse_tree_to_string(tree_elem):
    if isinstance(tree_elem, tokenize.Token):
        return repr(tree_elem.str)
    else:
        assert isinstance(tree_elem, TreeNode)
        return '%s(%s)' % (tree_elem.type, ','.join(map(parse_tree_to_string, tree_elem.child_nodes)))


def untested(e = None):
    raise Exception('untested')


# TODO: is IF_FOR needed?

PR_EXPR = 0
PL_EQ = 0
PR_SEMI = 0
PR_CLOSING = 0
PL_LAMBDA = 0
PR_IF_FOR = 1
PR_FOR = PR_IF_FOR                    # (x for x in x) if x
PR_IN_FOR = PR_FOR + 1                # x for (x in x)
PR_EQ = PR_IN_FOR + 1                 # Not strictly necessary.
PR_COMMA = PR_EQ + 1                  # x = (x , x).  By extension of the above, x for x in (x , x)
PR_AS = PR_COMMA + 1                  # (x as x) , x.  Also handles =(keyword) and *(unary).
PR_COLON = PR_AS + 1                  # (lambda x : x) as x
PL_ELSE = PR_COLON + 1                # x [ (x if x else x) : x ]
PR_IF = PL_ELSE + 1                   # x if x else (x if x else x).
PR_OR = PR_IF + 1                     # if (x or x)
PR_AND = PR_OR + 1                    # x or (x and x)
PR_U_NOT = PR_AND + 1                 # (not x) and x
PR_COMPARISON = PR_U_NOT + 1          # not (x < x)
PR_PIPE = PR_COMPARISON + 1           # (x | x) < x
PR_CARET = PR_PIPE + 1                # (x ^ x) | x
PR_AMPERSAND = PR_CARET + 1           # x ^ (x & x)
PR_SHIFT = PR_AMPERSAND + 1           # x & (x << x)
PR_PLUS = PR_SHIFT + 1                # x << (x + x)
PR_TIMES = PR_PLUS + 1                # x + (x * x)
PR_U_MINUS = PR_TIMES + 1             # (- x) * x
PR_POWER = PR_U_MINUS + 1             # - (x ** x)
PR_OPENING = PR_POWER + 1             # x ** (x ( ))
PR_DOT = PR_OPENING + 1               # (x . x) [ x ]
PR_MAX = 1000


POSTFIX_PRECEDENCES = {'.': PR_DOT,
                       '[': PR_OPENING,
                       '(': PR_OPENING,
                       '**': PR_POWER,
                       '*': PR_TIMES,
                       '//': PR_TIMES,
                       '/': PR_TIMES,
                       '%': PR_TIMES,
                       '+': PR_PLUS,
                       '-': PR_PLUS,
                       '<<': PR_SHIFT,
                       '>>': PR_SHIFT,
                       '&': PR_AMPERSAND,
                       '^': PR_CARET,
                       '|': PR_PIPE,
                       '<': PR_COMPARISON,
                       '>': PR_COMPARISON,
                       '==': PR_COMPARISON,
                       '>=': PR_COMPARISON,
                       '<=': PR_COMPARISON,
                       '<>': PR_COMPARISON,
                       '!=': PR_COMPARISON,
                       'is': PR_COMPARISON,
                       'in': PR_COMPARISON,
                       'in(for)': PR_IN_FOR,
                       'if(for)': PR_IF_FOR,
                       'not': PR_COMPARISON,
                       'and': PR_AND,
                       'or': PR_OR,
                       'if': PR_IF,
                       'as': PR_AS,
                       '=': PR_EQ,
                       '=(keyword)': PR_AS,
                       ':': PR_COLON,
                       'for': PR_FOR,
                       ',': PR_COMMA,
                       ';': PR_SEMI,
                       }

PREFIX_PRECEDENCES = dict(POSTFIX_PRECEDENCES)
for type in ['identifier', 'string', 'number', '(', '[', 'not', 'lambda', '{', '`', '...', '-', '+', '~', '.', '*',
             '**', ':', 'yield']:
    PREFIX_PRECEDENCES[type] = PR_MAX
for type in [')', ']', '}']:
    PREFIX_PRECEDENCES[type] = PR_CLOSING


class Context(object):
    '''Context of parsing.  This class contains the logic to decide what
    types of operations should be allowed when parsing an expression.
    '''
    def interpret_token(self, token):
        if token.type in self.token_substitutions:
            return self.token_substitutions[token.type]
        else:
            return token.type

    def determine_precedence(self, type):
        '''Return the precedence level of the given operator, in this
        context.
        '''

    def is_allowed_postfix(self, type):
        '''Find out whether the given token type is allowed as a postfix
        operator (i.e. after an expression).
        '''
        if type in self.postfix_exceptions:
            return False
        if type not in POSTFIX_PRECEDENCES:
            raise Exception('Unknown postfix precedence for token type %r' % type)
        return POSTFIX_PRECEDENCES[type] > self.threshold

    def is_allowed_prefix(self, type):
        '''Find out whether the given token type is allowed as a prefix
        operator (i.e. starting an expression).
        '''
        if type in self.prefix_exceptions:
            return False
        if type not in PREFIX_PRECEDENCES:
            raise Exception('Unknown prefix precedence for token type %r' % type)
        return PREFIX_PRECEDENCES[type] > self.threshold

    def minus(self, new_exception, postfix_only = False):
        '''Return a new Context that is the same as this one, but doesn't
        allow the given exception.

        Optional argument postfix_only, if true, means that the
        exception should only be in postfix context.
        '''
        result = Context()
        if postfix_only:
            result.prefix_exceptions = self.prefix_exceptions
        else:
            result.prefix_exceptions = self.prefix_exceptions | set([new_exception])
        result.postfix_exceptions = self.postfix_exceptions | set([new_exception])
        result.threshold = self.threshold
        result.token_substitutions = self.token_substitutions
        return result

    def gt(self, new_threshold, token_substitutions = None):
        '''Return a new Context that is the same as this one, but doesn't
        allow operators with precedence less than or equal to
        new_threshold.
        '''
        result = Context()
        result.prefix_exceptions = self.prefix_exceptions
        result.postfix_exceptions = self.postfix_exceptions
        result.threshold = max(self.threshold, new_threshold)
        if token_substitutions is None:
            result.token_substitutions = self.token_substitutions
        else:
            result.token_substitutions = dict(self.token_substitutions)
            result.token_substitutions.update(token_substitutions)
        return result

Context.top = Context()
Context.top.prefix_exceptions = set()
Context.top.postfix_exceptions = set()
Context.top.threshold = PR_EXPR
Context.top.token_substitutions = {}


class Parser(object):
    def __init__(self, tokens):
        self._tokens = tokens
        self._token_index = 0
        self._tree = TreeNode('root')
        self._stack = [self._tree]

    @property
    def result(self):
        assert len(self._stack) == 1
        assert len(self._stack[0].child_nodes) == 1
        return self._stack[0].child_nodes[0]

    @property
    def partial_result(self):
        if len(self._stack[0].child_nodes) == 1:
            return self._stack[0].child_nodes[0]
        else:
            return None

    @property
    def current_token(self):
        if self._token_index >= len(self._tokens):
            return tokenize.Token('end', '')
        return self._tokens[self._token_index]

    def advance(self, expected_token_type = None):
        if expected_token_type is not None and self.current_token.type != expected_token_type:
            raise Exception('At %r: expected %r' % (self.current_token.str, expected_token_type))
        self._stack[-1].append(self.current_token)
        self._token_index += 1

    def open(self, type):
        node = TreeNode(type)
        self._stack[-1].append(node)
        self._stack.append(node)

    def open_insert(self, type):
        node = TreeNode(type, self._stack[-1].de_append())
        self._stack[-1].append(node)
        self._stack.append(node)

    def close(self):
        del self._stack[-1]

    def make_expr_handler(type, *constituents):
        def parser(self, context):
            self.open(type)
            self.advance()
            for constituent in constituents:
                constituent(self, context)
            self.close()
        return parser

    def make_match_handler(contents, matching_token, postfix_only):
        def parser(self, context):
            self.open('matched')
            self.advance()
            contents(self, Context.top.minus(matching_token, postfix_only))
            self.advance(matching_token)
            self.close()
        return parser

    def exact_token(type):
        def parser(self, context):
            self.advance(type)
        return parser

    def expression(self, context):
        if not context.is_allowed_prefix(context.interpret_token(self.current_token)):
            return
        if not context.interpret_token(self.current_token) in Parser.EXPR_STARTERS:
            raise Exception('At %r: cannot start expression with %r' % (self.current_token.str, context.interpret_token(self.current_token)))
        Parser.EXPR_STARTERS[context.interpret_token(self.current_token)](self, context)
        while context.interpret_token(self.current_token) in Parser.EXPR_CONTINUATIONS:
            if not context.is_allowed_postfix(context.interpret_token(self.current_token)):
                break
            Parser.EXPR_CONTINUATIONS[context.interpret_token(self.current_token)](self, context)

    ASSIGNMENT_OPS = ['=', '+=', '-=', '*=', '/=', '%=', '**=', '>>=', '<<=', '&=', '^=', '|=']

    def assignment_rhs_handler(self, context):
        self.open_insert('binary')
        while context.interpret_token(self.current_token) in Parser.ASSIGNMENT_OPS:
            self.advance()
            if context.interpret_token(self.current_token) == 'yield':
                self.yield_expr(context)
            else:
                self.expression(context)
        self.close()

    def simple_stmt(self, context):
        if context.interpret_token(self.current_token) in Parser.STMT_STARTERS:
            Parser.STMT_STARTERS[context.interpret_token(self.current_token)](self, context)
        else:
            self.expression(context)
            if context.interpret_token(self.current_token) in Parser.ASSIGNMENT_OPS:
                self.assignment_rhs_handler(context)

    def stmt_list(self, context):
        self.open('stmt_list')
        while True:
            if context.interpret_token(self.current_token) == 'newline':
                break
            self.simple_stmt(context)
            if context.interpret_token(self.current_token) == 'newline':
                break
            if context.interpret_token(self.current_token) == ';':
                self.advance()
        self.advance('newline')
        self.close()

    def top(self):
        self.statement(Context.top)

    def statement(self, context):
        if context.interpret_token(self.current_token) in Parser.COMPOUND_STMT_STARTERS:
            Parser.COMPOUND_STMT_STARTERS[context.interpret_token(self.current_token)](self, context)
            if context.interpret_token(self.current_token) in Parser.COMPOUND_STMT_CONTINUATIONS:
                self.open_insert('compound_stmt')
                while context.interpret_token(self.current_token) in Parser.COMPOUND_STMT_CONTINUATIONS:
                    Parser.COMPOUND_STMT_CONTINUATIONS[context.interpret_token(self.current_token)](self, context)
                self.close()
        else:
            self.stmt_list(context)

    def yield_expr(self, context):
        self.advance()
        if context.interpret_token(self.current_token) in Parser.EXPR_STARTERS:
            self.open_insert('yield_expression')
            self.expression(context)
            self.close()

    def make_expression_parser(new_threshold):
        def parser(self, context):
            self.expression(context.gt(new_threshold))
        return parser

    def lambda_handler(self, context):
        self.open('lambda')
        self.advance()
        self.expression(Context.top.minus(':').gt(PL_LAMBDA))
        self.advance(':')
        self.expression(context.gt(PR_COLON))
        self.close()

    def single_token_handler(self, context):
        self.advance()

    def key_datum_list(self, context):
        if context.interpret_token(self.current_token) not in Parser.EXPR_STARTERS:
            return              # Empty list
        self.key_datum(context)
        if context.interpret_token(self.current_token) == ',':
            self.open_insert('key_datum_list')
            while context.interpret_token(self.current_token) == ',':
                self.advance()
                if context.interpret_token(self.current_token) in Parser.EXPR_STARTERS:
                    self.key_datum(context)
            self.close()

    def key_datum(self, context):
        self.open('key_datum')
        self.expression(context.minus(':'))
        self.advance(':')
        self.expression(context.gt(PR_COLON))
        self.close()

    def unary_slice_handler(self, context):
        self.advance()
        if context.interpret_token(self.current_token) in Parser.EXPR_STARTERS and context.interpret_token(self.current_token) != ':':
            self.open_insert('prefix')
            self.expression(context.gt(PR_COLON))
            self.close()

    def unary_dot_handler(self, context):
        self.advance()
        opened = False
        while context.interpret_token(self.current_token) == '.':
            if opened:
                self.close()
            self.open_insert('binary')
            self.advance()
            opened = True
        if context.interpret_token(self.current_token) in Parser.EXPR_STARTERS:
            if not opened:
                self.open_insert('prefix')
            self.expression(context.gt(PR_DOT))
            opened = True
        if opened:
            self.close()

    EXPR_STARTERS = {'identifier': single_token_handler,
                     'string': single_token_handler,
                     'number': single_token_handler,
                     '(': make_match_handler(make_expression_parser(PR_EXPR), ')', False),
                     '[': make_match_handler(make_expression_parser(PR_EXPR), ']', False),
                     'not': make_expr_handler('prefix', make_expression_parser(PR_U_NOT)),
                     'lambda': lambda_handler,
                     '{': make_expr_handler('matched', key_datum_list, exact_token('}')),
                     '`': make_match_handler(make_expression_parser(PR_EXPR), '`', True),
                     '...': single_token_handler,
                     '-': make_expr_handler('prefix', make_expression_parser(PR_U_MINUS)),
                     '+': make_expr_handler('prefix', make_expression_parser(PR_U_MINUS)),
                     '~': make_expr_handler('prefix', make_expression_parser(PR_U_MINUS)),
                     '.': unary_dot_handler,
                     '*': make_expr_handler('prefix', make_expression_parser(PR_AS)),
                     '**': make_expr_handler('prefix', make_expression_parser(PR_AS)),
                     ':': unary_slice_handler,
                     'yield': yield_expr,
                     }

    def binary(new_threshold, token_substitutions = None):
        def parser(self, context):
            type = context.interpret_token(self.current_token)
            self.open_insert('binary')
            self.advance()
            self.expression(context.gt(new_threshold, token_substitutions))
            self.close()
        return parser

    def is_handler(self, context):
        self.open_insert('binary')
        self.advance()
        # TODO: "is not" and "not in" should be grouped.
        if context.interpret_token(self.current_token) == 'not':
            self.advance()
        self.expression(context.gt(PR_COMPARISON))
        self.close()

    def not_handler(self, context):
        self.open_insert('binary')
        self.advance()
        self.advance('in')
        self.expression(context.gt(PR_COMPARISON))
        self.close()

    def if_handler(self, context):
        self.open_insert('binary')
        self.advance()
        self.expression(context.minus('else').gt(PR_IF))
        self.close()
        if self.current_token.type == 'else':
            self.open_insert('binary')
            self.advance('else')
            self.expression(context.gt(PL_ELSE))
            self.close()

    def subscription_handler(self, context):
        self.open_insert('subscription')
        self.advance()
        self.expression(Context.top.minus(']'))
        self.advance(']')
        self.close()

    def slice_rhs_handler(self, context):
        self.open_insert('binary')
        self.advance()
        if context.interpret_token(self.current_token) in Parser.EXPR_STARTERS and context.interpret_token(self.current_token) != ':':
            self.expression(context.gt(PR_COLON))
        self.close()

    def call_handler(self, context):
        self.open_insert('call')
        self.advance()
        self.expression(Context.top.minus(')').gt(PR_EXPR, token_substitutions = {'=': '=(keyword)'}))
        self.advance(')')
        self.close()

    EXPR_CONTINUATIONS = {'.': binary(PR_DOT),
                          '[': subscription_handler,
                          '(': call_handler,
                          '**': binary(PR_POWER - 1),
                          '*': binary(PR_TIMES),
                          '//': binary(PR_TIMES),
                          '/': binary(PR_TIMES),
                          '%': binary(PR_TIMES),
                          '+': binary(PR_PLUS),
                          '-': binary(PR_PLUS),
                          '<<': binary(PR_SHIFT),
                          '>>': binary(PR_SHIFT),
                          '&': binary(PR_AMPERSAND),
                          '^': binary(PR_CARET),
                          '|': binary(PR_PIPE),
                          '<': binary(PR_COMPARISON),
                          '>': binary(PR_COMPARISON),
                          '==': binary(PR_COMPARISON),
                          '>=': binary(PR_COMPARISON),
                          '<=': binary(PR_COMPARISON),
                          '<>': binary(PR_COMPARISON),
                          '!=': binary(PR_COMPARISON),
                          'is': is_handler,
                          'in': binary(PR_COMPARISON),
                          'in(for)': binary(PR_IN_FOR),
                          'if(for)': binary(PR_IF_FOR),
                          'not': not_handler,
                          'and': binary(PR_AND),
                          'or': binary(PR_OR),
                          'if': if_handler,
                          'as': binary(PR_AS),
                          '=': binary(PR_EQ - 1),
                          '=(keyword)': binary(PR_AS - 1),
                          ':': slice_rhs_handler,
                          'for': binary(PR_FOR, token_substitutions = {'in': 'in(for)', 'if': 'if(for)'}),
                          ',': binary(PR_COMMA),
                          }

    def make_stmt_handler(type, *constituents):
        def parser(self, context):
            self.open(type)
            self.advance()
            for constituent in constituents:
                constituent(self, context)
            self.close()
        return parser

    def print_handler(self, context):
        self.open('print')
        self.advance()
        if context.interpret_token(self.current_token) == '>>':
            self.advance()
        self.expression(context)
        self.close()

    def exec_handler(self, context):
        self.open('exec')
        self.advance()
        self.expression(context.minus('in'))
        if context.interpret_token(self.current_token) == 'in':
            self.advance()
            self.expression(context)
        self.close()

    def from_handler(self, context):
        self.open('import')
        self.advance()
        self.expression(context.minus('import'))
        self.advance('import')
        if context.interpret_token(self.current_token) == '*':
            self.advance()
        else:
            self.expression(context)
        self.close()

    def consume_single_token(self, context):
        self.advance()

    STMT_STARTERS = {'assert': make_stmt_handler('assert', expression),
                     'pass': consume_single_token,
                     'del': make_stmt_handler('del', expression),
                     'print': print_handler,
                     'return': make_stmt_handler('return', expression),
                     'raise': make_stmt_handler('raise', expression),
                     'break': make_stmt_handler('break'),
                     'continue': make_stmt_handler('continue'),
                     'import': make_stmt_handler('import', expression),
                     'global': make_stmt_handler('global', expression),
                     'exec': exec_handler,
                     'from': from_handler,
                     }

    def suite(self, context):
        if context.interpret_token(self.current_token) == 'newline':
            self.open('suite')
            self.advance()
            self.advance('indent')
            while context.interpret_token(self.current_token) != 'dedent':
                self.statement(context)
            self.advance('dedent')
            self.close()
        else:
            self.stmt_list(context)

    def for_handler(self, context):
        self.expression(context.gt(PR_FOR, token_substitutions = {'in': 'in(for)'}));
        self.expression(context)

    def do_nothing(self, context):
        pass

    def def_handler(self, context):
        self.advance()
        self.advance('(')
        self.expression(context.gt(PR_EXPR, token_substitutions = {'=': '=(keyword)'}))
        self.advance(')')

    def decorator_handler(self, context):
        self.open('decorated')
        self.open('decorator')
        self.advance()
        self.expression(context)
        self.advance('newline')
        self.close()
        self.statement(context)
        self.close()

    def make_clause_parser(contents_parser):
        def parser(self, context):
            self.open(context.interpret_token(self.current_token))
            self.advance()
            contents_parser(self, context.minus(':'))
            self.advance(':')
            self.suite(context)
            self.close()
        return parser

    COMPOUND_STMT_STARTERS = {'if': make_clause_parser(make_expression_parser(PR_EXPR)),
                              'while': make_clause_parser(make_expression_parser(PR_EXPR)),
                              'for': make_clause_parser(for_handler),
                              'try': make_clause_parser(do_nothing),
                              'with': make_clause_parser(make_expression_parser(PR_EXPR)),
                              'def': make_clause_parser(def_handler),
                              'class': make_clause_parser(make_expression_parser(PR_EXPR)),
                              '@': decorator_handler,
                              }

    COMPOUND_STMT_CONTINUATIONS = {'except': make_clause_parser(expression),
                                   'elif': make_clause_parser(make_expression_parser(PR_EXPR)),
                                   'else': make_clause_parser(do_nothing),
                                   'finally': make_clause_parser(do_nothing),
                                   }

# TODO: make sure (a for b in c, d) gives a syntax error.

