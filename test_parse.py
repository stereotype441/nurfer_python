import unittest
import sys
import functools
import random

import tokenize
import parse


def Token(t):
    return ('token', t)


def Either(*alts):
    return ('either',) + alts


def Sequence(tree_node_type, *elems, **kwargs):
    if kwargs.keys() == ['penalty']:
        penalty = kwargs['penalty']
    else:
        penalty = 1
    assert len(elems) >= 1
    return ('sequence', tree_node_type, penalty) + elems


def Symbol(str):
    return Token(tokenize.Token(str, str))


grammar_rules = {'start': 'statement',
                 'identifier': Token(tokenize.Token('identifier', 'x')),
                 'stringliteral': Token(tokenize.Token('string', "'s'")),
                 'number': Token(tokenize.Token('number', '3')),
                 'atom': Either('identifier', 'literal', 'enclosure'),
                 'enclosure': Either('parenth_form', 'list_display', 'generator_expression', 'dict_display',
                                     'string_conversion', 'yield_atom'),
                 'literal': Either('stringliteral', 'number'),
                 'parenth_form': Either(Sequence('matched', Symbol('('), Symbol(')')),
                                        Sequence('matched', Symbol('('), 'expression_list', Symbol(')'))),
                 'list_display': Either(Sequence('matched', Symbol('['), Symbol(']')),
                                        Sequence('matched', Symbol('['), Either('expression_list', 'list_comprehension'),
                                                 Symbol(']'))),
                 'list_comprehension': Either(Sequence('binary', Either('expression', 'list_comprehension'),
                                                       Symbol('for'), 'for_spec'),
                                              Sequence('binary', 'list_comprehension', Symbol('if'), 'old_expression')),
                 'for_spec': Sequence('binary', 'target_list', Symbol('in'), 'old_expression_list'),
                 'old_expression_list': Either('old_expression',
                                               Sequence('binary', 'old_expression', Symbol(','), 'old_expression'),
                                               Sequence('binary',
                                                        Sequence('binary', 'old_expression', Symbol(','),
                                                                 'old_expression'),
                                                        Symbol(','))),
                 'generator_expression': Sequence('matched', Symbol('('), 'generator_expression2', Symbol(')')),
                 'generator_expression2': Either(Sequence('binary', Either('expression', 'generator_expression2'), Symbol('for'), 'generator_for_spec'),
                                                 Sequence('binary', 'generator_expression2', Symbol('if'), 'old_expression')),
                 'generator_for_spec': Sequence('binary', 'target_list', Symbol('in'), 'or_test'),
                 'dict_display': Either(Sequence('matched', Symbol('{'), Symbol('}')),
                                        Sequence('matched', Symbol('{'), 'key_datum_list', Symbol('}'))),
                 'key_datum_list': Either('key_datum',
                                          Sequence('key_datum_list', 'key_datum', Symbol(',')),
                                          Sequence('key_datum_list', 'key_datum', Symbol(','), 'key_datum'),
                                          Sequence('key_datum_list', 'key_datum', Symbol(','), 'key_datum', Symbol(','))),
                 'key_datum': Sequence('key_datum', 'expression', Symbol(':'), 'expression', penalty=0),
                 'string_conversion': Sequence('matched', Symbol('`'), 'expression_list_proper', Symbol('`')),
                 'yield_atom': Sequence('matched', Symbol('('), 'yield_expression', Symbol(')')),
                 'yield_expression': Either(Symbol('yield'),
                                            Sequence('yield_expression', Symbol('yield'), 'expression_list', penalty=0)),
                 'primary': Either('atom', 'attributeref', 'subscription', 'slicing', 'call'),
                 'attributeref': Sequence('binary', 'primary', Symbol('.'), 'identifier'),
                 'subscription': Sequence('subscription', 'primary', Symbol('['), 'expression_list', Symbol(']')),
                 'slicing': Either('simple_slicing', 'extended_slicing'),
                 'simple_slicing': Sequence('subscription', 'primary', Symbol('['), 'short_slice', Symbol(']')),
                 'extended_slicing': Sequence('subscription', 'primary', Symbol('['), 'slice_list', Symbol(']')),
                 'slice_list': Either('slice_item', Sequence('binary', 'slice_item', Symbol(',')),
                                      Sequence('binary', 'slice_item', Symbol(','), 'slice_item'),
                                      Sequence('binary', Sequence('binary', 'slice_item', Symbol(','), 'slice_item'),
                                               Symbol(','))),
                 'slice_item': Either('expression', 'proper_slice', 'ellipsis'),
                 'proper_slice': Either('short_slice', 'long_slice'),
                 'short_slice': Either(Symbol(':'),
                                       Sequence('prefix', Symbol(':'), 'expression'),
                                       Sequence('binary', 'expression', Symbol(':')),
                                       Sequence('binary', 'expression', Symbol(':'), 'expression')),
                 'long_slice': Either(Sequence('binary', Symbol(':'), Symbol(':'), 'expression'),
                                       Sequence('binary', Sequence('prefix', Symbol(':'), 'expression'), Symbol(':'), 'expression'),
                                       Sequence('binary', Sequence('binary', 'expression', Symbol(':')), Symbol(':'), 'expression'),
                                       Sequence('binary', Sequence('binary', 'expression', Symbol(':'), 'expression'), Symbol(':'), 'expression')),
                 'ellipsis': Symbol('...'),
                 'call': Either(Sequence('call', 'primary', Symbol('('), Symbol(')')),
                                Sequence('call', 'primary', Symbol('('), 'argument_list', Symbol(')')),
                                Sequence('call', 'primary', Symbol('('), 'generator_expression2', Symbol(')'))),
                 'argument_list': Either('argument',
                                         Sequence('binary', 'argument', Symbol(',')),
                                         Sequence('binary', 'argument', Symbol(','), 'argument'),
                                         Sequence('binary', Sequence('binary', 'argument', Symbol(','), 'argument'),
                                                  Symbol(','))),
                 'argument': Either('expression', Sequence('prefix', Symbol('*'), 'expression'), Sequence('prefix', Symbol('**'), 'expression'), 'keyword_item'),
                 'keyword_item': Sequence('binary', 'identifier', Symbol('='), 'expression'),
                 'power': Either('primary', Sequence('binary', 'primary', Symbol('**'), 'u_expr')),
                 'u_expr': Either('power', Sequence('prefix', Either(Symbol('-'), Symbol('+'), Symbol('~')), 'u_expr')),
                 'm_expr': Either('u_expr', Sequence('binary', 'm_expr', Either(Symbol('*'), Symbol('//'), Symbol('/'), Symbol('%')), 'u_expr')),
                 'a_expr': Either('m_expr', Sequence('binary', 'a_expr', Either(Symbol('+'), Symbol('-')), 'm_expr')),
                 'shift_expr': Either('a_expr', Sequence('binary', 'shift_expr', Either(Symbol('<<'), Symbol('>>')), 'a_expr')),
                 'and_expr': Either('shift_expr', Sequence('binary', 'and_expr', Symbol('&'), 'shift_expr')),
                 'xor_expr': Either('and_expr', Sequence('binary', 'xor_expr', Symbol('^'), 'and_expr')),
                 'or_expr': Either('xor_expr', Sequence('binary', 'or_expr', Symbol('|'), 'xor_expr')),
                 'comparison': Either('or_expr',
                                      Sequence('binary', 'comparison',
                                               Either(*[Symbol(s) for s in ['<', '>', '==', '>=', '<=', '<>', '!=',
                                                                            'is', 'in']]),
                                               'or_expr'),
                                      Sequence('binary', 'comparison', Symbol('is'), Symbol('not'), 'or_expr'),
                                      Sequence('binary', 'comparison', Symbol('not'), Symbol('in'), 'or_expr')),
                 'expression': Either('conditional_expression', 'lambda_form'),
                 'old_expression': Either('or_test', 'old_lambda_form'),
                 'conditional_expression': Either('or_test',
                                                  Sequence('binary',
                                                           Sequence('binary', 'or_test', Symbol('if'), 'or_test'),
                                                           Symbol('else'), 'expression')),
                 'or_test': Either('and_test', Sequence('binary', 'or_test', Symbol('or'), 'and_test')),
                 'and_test': Either('not_test', Sequence('binary', 'and_test', Symbol('and'), 'not_test')),
                 'not_test': Either('comparison', Sequence('prefix', Symbol('not'), 'not_test')),
                 'lambda_form': Either(Sequence('lambda', Symbol('lambda'), 'parameter_list', Symbol(':'), 'expression'),
                                       Sequence('lambda', Symbol('lambda'), Symbol(':'), 'expression')),
                 'old_lambda_form': Either(Sequence('lambda', Symbol('lambda'), 'parameter_list', Symbol(':'), 'old_expression'),
                                           Sequence('lambda', Symbol('lambda'), Symbol(':'), 'old_expression')),
                 'expression_list': Either('expression',
                                           Sequence('binary', 'expression', Symbol(',')),
                                           Sequence('binary', 'expression', Symbol(','), 'expression'),
                                           Sequence('binary',
                                                    Sequence('binary', 'expression', Symbol(','), 'expression'),
                                                    Symbol(','))),
                 'expression_list_proper': Either('expression',
                                                  Sequence('binary', 'expression', Symbol(','), 'expression')),
                 'simple_stmt': Either('expression_stmt', 'assert_stmt', 'assignment_stmt', 'augmented_assignment_stmt',
                                       'pass_stmt', 'del_stmt', 'print_stmt', 'return_stmt', 'yield_stmt', 'raise_stmt',
                                       'break_stmt', 'continue_stmt', 'import_stmt', 'global_stmt', 'exec_stmt'),
                 'expression_stmt': 'expression_list',
                 'assert_stmt': Sequence('assert', Symbol('assert'),
                                         Either('expression',
                                                Sequence('binary', 'expression', Symbol(','), 'expression'))),
                 'assignment_stmt': Either(Sequence('binary', 'target_list', Symbol('='), 'expression_list'),
                                           Sequence('binary', 'target_list', Symbol('='), 'yield_expression'),
                                           Sequence('binary', 'target_list', Symbol('='),
                                                    Sequence('binary', 'target_list', Symbol('='), 'expression_list')),
                                           Sequence('binary', 'target_list', Symbol('='),
                                                    Sequence('binary', 'target_list', Symbol('='), 'yield_expression'))),
                 'target_list': Either('target',
                                       Sequence('binary', 'target', Symbol(',')),
                                       Sequence('binary', 'target', Symbol(','), 'target'),
                                       Sequence('binary', Sequence('binary', 'target', Symbol(','), 'target'),
                                                Symbol(','))),
                 'target': Either('identifier', Sequence('matched', Symbol('('), 'target_list', Symbol(')')),
                                  Sequence('matched', Symbol('['), 'target_list', Symbol(']')), 'attributeref',
                                  'subscription', 'slicing'),
                 'augmented_assignment_stmt': Either(Sequence('binary', 'target', 'augop', 'expression_list'),
                                                     Sequence('binary', 'target', 'augop', 'yield_expression')),
                 'augop': Either(*map(Symbol, ['+=', '-=', '*=', '/=', '%=', '**=', '>>=', '<<=', '&=', '^=', '|='])),
                 'pass_stmt': Symbol('pass'),
                 'del_stmt': Sequence('del', Symbol('del'), 'target_list'),
                 'print_stmt': Either(Sequence('print', Symbol('print')),
                                      Sequence('print', Symbol('print'), 'expression'),
                                      Sequence('print', Symbol('print'),
                                               Sequence('binary', 'expression', Symbol(','))),
                                      Sequence('print', Symbol('print'),
                                               Sequence('binary', 'expression', Symbol(','), 'expression')),
                                      Sequence('print', Symbol('print'),
                                               Sequence('binary',
                                                        Sequence('binary', 'expression', Symbol(','), 'expression'),
                                                        Symbol(','))),
                                      Sequence('print', Symbol('print'), Symbol('>>'), 'expression'),
                                      Sequence('print', Symbol('print'), Symbol('>>'),
                                               Sequence('binary', 'expression', Symbol(','))),
                                      Sequence('print', Symbol('print'), Symbol('>>'),
                                               Sequence('binary', 'expression', Symbol(','), 'expression')),
                                      Sequence('print', Symbol('print'), Symbol('>>'),
                                               Sequence('binary',
                                                        Sequence('binary', 'expression', Symbol(','), 'expression'),
                                                        Symbol(',')))),
                 'return_stmt': Either(Sequence('return', Symbol('return')),
                                       Sequence('return', Symbol('return'), 'expression_list')),
                 'yield_stmt': 'yield_expression',
                 'raise_stmt': Either(Sequence('raise', Symbol('raise')),
                                      Sequence('raise', Symbol('raise'), 'expression'),
                                      Sequence('raise', Symbol('raise'),
                                               Sequence('binary', 'expression', Symbol(','), 'expression')),
                                      Sequence('raise', Symbol('raise'),
                                               Sequence('binary',
                                                        Sequence('binary', 'expression', Symbol(','), 'expression'),
                                                        Symbol(','), 'expression'))),
                 'break_stmt': Sequence('break', Symbol('break')),
                 'continue_stmt': Sequence('continue', Symbol('continue')),
                 'import_stmt': Either(Sequence('import', Symbol('import'), 'module_as'),
                                       Sequence('import', Symbol('import'),
                                                Sequence('binary', 'module_as', Symbol(','), 'module_as')),
                                       Sequence('import', Symbol('from'), 'relative_module', Symbol('import'),
                                                'identifier_as'),
                                       Sequence('import', Symbol('from'), 'relative_module', Symbol('import'),
                                                Sequence('binary', 'identifier_as', Symbol(','), 'identifier_as')),
                                       Sequence('import', Symbol('from'), 'relative_module', Symbol('import'),
                                                Sequence('matched', Symbol('('), 'identifier_as', Symbol(')'))),
                                       Sequence('import', Symbol('from'), 'relative_module', Symbol('import'),
                                                Sequence('matched', Symbol('('),
                                                         Sequence('binary', 'identifier_as', Symbol(',')),
                                                         Symbol(')'))),
                                       Sequence('import', Symbol('from'), 'relative_module', Symbol('import'),
                                                Sequence('matched', Symbol('('),
                                                         Sequence('binary', 'identifier_as', Symbol(','),
                                                                  'identifier_as'),
                                                         Symbol(')'))),
                                       Sequence('import', Symbol('from'), 'relative_module', Symbol('import'),
                                                Sequence('matched', Symbol('('),
                                                         Sequence('binary',
                                                                  Sequence('binary', 'identifier_as', Symbol(','),
                                                                           'identifier_as'),
                                                                  Symbol(',')),
                                                         Symbol(')'))),
                                       Sequence('import', Symbol('from'), 'relative_module', Symbol('import'),
                                                Symbol('*'))),
                 'module_as': Either('module', Sequence('binary', 'module', Symbol('as'), 'name')),
                 'identifier_as': Either('identifier', Sequence('binary', 'identifier', Symbol('as'), 'name')),
                 'module': Either('identifier', Sequence('binary', 'module', Symbol('.'), 'identifier')),
                 'relative_module': Either('module',
                                           Symbol('.'),
                                           Sequence('binary', Symbol('.'), Symbol('.')),
                                           Sequence('binary',
                                                    Sequence('binary', Symbol('.'), Symbol('.')), Symbol('.')),
                                           Sequence('prefix', Symbol('.'), 'identifier'),
                                           Sequence('binary', Symbol('.'), Symbol('.'), 'identifier'),
                                           Sequence('binary', Sequence('prefix', Symbol('.'), 'identifier'),
                                                    Symbol('.'), 'identifier'),
                                           Sequence('binary',
                                                    Sequence('binary', Symbol('.'), Symbol('.'), 'identifier'),
                                                    Symbol('.'), 'identifier')),
                 'name': 'identifier',
                 'global_stmt': Either(Sequence('global', Symbol('global'), 'identifier'),
                                       Sequence('global', Symbol('global'),
                                                Sequence('binary', 'identifier', Symbol(','), 'identifier'))),
                 'exec_stmt': Either(Sequence('exec', Symbol('exec'), 'or_expr'),
                                     Sequence('exec', Symbol('exec'), 'or_expr', Symbol('in'), 'expression'),
                                     Sequence('exec', Symbol('exec'), 'or_expr', Symbol('in'),
                                              Sequence('binary', 'expression', Symbol(','), 'expression'))),
                 'compound_stmt': Either('if_stmt', 'while_stmt', 'for_stmt', 'try_stmt', 'with_stmt', 'funcdef',
                                         'classdef'),
                 'suite': Either('stmt_list',
                                 Sequence('suite', Symbol('newline'), Symbol('indent'), 'statement', Symbol('dedent')),
                                 Sequence('suite', Symbol('newline'), Symbol('indent'), 'statement', 'statement',
                                          Symbol('dedent'))),
                 'statement': Either('stmt_list', 'compound_stmt'),
                 'stmt_list': Sequence('stmt_list', 'simple_stmt', Symbol('newline')),
                 'stmt_list': Sequence('stmt_list', 'simple_stmt', Symbol(';'), Symbol('newline')),
                 'stmt_list': Sequence('stmt_list', 'simple_stmt', Symbol(';'), 'simple_stmt', Symbol('newline')),
                 'stmt_list': Sequence('stmt_list', 'simple_stmt', Symbol(';'), 'simple_stmt', Symbol(';'), Symbol('newline')),
                 'if_stmt': Either('if_clause',
                                   Sequence('compound_stmt', 'if_clause', 'else_clause'),
                                   Sequence('compound_stmt', 'if_clause', 'elif_clause'),
                                   Sequence('compound_stmt', 'if_clause', 'elif_clause', 'else_clause'),
                                   Sequence('compound_stmt', 'if_clause', 'elif_clause', 'elif_clause'),
                                   Sequence('compound_stmt', 'if_clause', 'elif_clause', 'elif_clause', 'else_clause')),
                 'if_clause': Sequence('if', Symbol('if'), 'expression', Symbol(':'), 'suite'),
                 'else_clause': Sequence('else', Symbol('else'), Symbol(':'), 'suite'),
                 'elif_clause': Sequence('elif', Symbol('elif'), 'expression', Symbol(':'), 'suite'),
                 'while_stmt': Either('while_clause', Sequence('compound_stmt', 'while_clause', 'else_clause')),
                 'while_clause': Sequence('while', Symbol('while'), 'expression', Symbol(':'), 'suite'),
                 'for_stmt': Either('for_clause', Sequence('compound_stmt', 'for_clause', 'else_clause')),
                 'for_clause': Sequence('for', Symbol('for'),
                                        Sequence('binary', 'target_list', Symbol('in'), 'expression_list'), Symbol(':'),
                                        'suite'),
                 'try_stmt': Either('try1_stmt', 'try2_stmt'),
                 'try1_stmt': Either(Sequence('compound_stmt', 'try_clause', 'except_clause'),
                                     Sequence('compound_stmt', 'try_clause', 'except_clause', 'except_clause'),
                                     Sequence('compound_stmt', 'try_clause', 'except_clause', 'else_clause'),
                                     Sequence('compound_stmt', 'try_clause', 'except_clause', 'except_clause',
                                              'else_clause'),
                                     Sequence('compound_stmt', 'try_clause', 'except_clause', 'finally_clause'),
                                     Sequence('compound_stmt', 'try_clause', 'except_clause', 'except_clause',
                                              'finally_clause'),
                                     Sequence('compound_stmt', 'try_clause', 'except_clause', 'else_clause',
                                              'finally_clause'),
                                     Sequence('compound_stmt', 'try_clause', 'except_clause', 'except_clause',
                                              'else_clause', 'finally_clause')),
                 'try_clause': Sequence('try', Symbol('try'), Symbol(':'), 'suite'),
                 'except_clause': Either(Sequence('except', Symbol('except'), Symbol(':'), 'suite'),
                                         Sequence('except', Symbol('except'), 'expression', Symbol(':'), 'suite'),
                                         Sequence('except', Symbol('except'),
                                                  Sequence('binary', 'expression', Symbol(','), 'target'),
                                                  Symbol(':'), 'suite')),
                 'try2_stmt': Sequence('compound_stmt', 'try_clause', 'finally_clause'),
                 'finally_clause': Sequence('finally', Symbol('finally'), Symbol(':'), 'suite'),
                 'with_stmt': Either(Sequence('with', Symbol('with'), 'expression', Symbol(':'), 'suite'),
                                     Sequence('with', Symbol('with'),
                                              Sequence('binary', 'expression', Symbol('as'), 'target'), Symbol(':'),
                                              'suite')),
                 'funcdef': Either(Sequence('def', Symbol('def'), 'funcname', Symbol('('), 'parameter_list',
                                            Symbol(')'), Symbol(':'), 'suite'),
                                   Sequence('decorated', 'decorator', 'funcdef')),
                 'decorator': Either(Sequence('decorator', Symbol('@'), 'dotted_name', Symbol('newline')),
                                     Sequence('decorator', Symbol('@'),
                                              Sequence('call', 'dotted_name', Symbol('('), Symbol(')')),
                                              Symbol('newline')),
                                     Sequence('decorator', Symbol('@'),
                                              Sequence('call', 'dotted_name', Symbol('('), 'argument_list',
                                                       Symbol(')')),
                                              Symbol('newline'))),
                 'dotted_name': Either('identifier', Sequence('binary', 'dotted_name', Symbol('.'), 'identifier')),
                 'parameter_list': Either('param', Sequence('binary', 'param', Symbol(',')),
                                          Sequence('binary', 'param', Symbol(','), 'param'),
                                          Sequence('binary', Sequence('binary', 'param', Symbol(','), 'param'),
                                                   Symbol(','))),
                 'param': Either('parameter', Sequence('prefix', Symbol('*'), 'identifier'),
                                 Sequence('prefix', Symbol('**'), 'identifier'),
                                 Sequence('binary', 'parameter', Symbol('='), 'expression')),
                 'sublist': Either('parameter',
                                   Sequence('binary', 'parameter', Symbol(',')),
                                   Sequence('binary', 'parameter', Symbol(','), 'parameter'),
                                   Sequence('binary', Sequence('binary', 'parameter', Symbol(','), 'parameter'),
                                            Symbol(','))),
                 'parameter': Either('identifier', Sequence('matched', Symbol('('), 'sublist', Symbol(')'))),
                 'funcname': 'identifier',
                 'classdef': Either(Sequence('class', Symbol('class'), 'classname', Symbol(':'), 'suite'),
                                    Sequence('class', Symbol('class'),
                                             Sequence('call', 'classname', Symbol('('), Symbol(')')), Symbol(':'),
                                             'suite'),
                                    Sequence('class', Symbol('class'),
                                             Sequence('call', 'classname', Symbol('('), 'expression_list', Symbol(')')),
                                             Symbol(':'), 'suite')),
                 'classname': 'identifier',
                 }


def for_each_rule(rules, f):
    """Call f(key, value) on each element of rules.  If f adds new keys to
    rules, f is guaranteed to be called on those keys.
    """
    visited_rules = set()
    while True:
        unvisited_rules = list(set(rules.keys()) - visited_rules)
        if len(unvisited_rules) == 0:
            return
        next_rule = unvisited_rules[0]
        f(next_rule, rules[next_rule])
        visited_rules.add(next_rule)


num_leading_params = {'either': 1, 'sequence': 3}


def normalize_rules(rules):
    new_rule_index = [0]
    def insert_new_rule(value):
        key = 'g_%r' % new_rule_index[0]
        new_rule_index[0] += 1
        rules[key] = value
        return key

    def make_mutable(key, value): # Transform tuples to lists
        if isinstance(value, basestring):
            return
        rules[key] = list(value)
    for_each_rule(rules, make_mutable)

    def flatten(key, value):
        if isinstance(value, basestring):
            return
        if value[0] == 'token':
            return
        for i in xrange(num_leading_params[value[0]], len(value)):
            if isinstance(value[i], basestring):
                while isinstance(rules[value[i]], basestring):
                    value[i] = rules[value[i]] # Get rid of long chains
            else:
                value[i] = insert_new_rule(list(value[i])) # Get rid of deep rules
    for_each_rule(rules, flatten)

    def expand_either(key, value):
        if isinstance(value, basestring):
            return
        if value[0] == 'either':
            i = num_leading_params[value[0]]
            while i < len(value):
                referenced_rule = rules[value[i]]
                assert not isinstance(referenced_rule, basestring) # Previous operation should have guaranteed this
                if referenced_rule[0] == 'either':
                    value[i : i + 1] = list(referenced_rule[num_leading_params[referenced_rule[0]] :])
                else:
                    i += 1
    for_each_rule(rules, expand_either)

    def follow_chains(key, value):
        while isinstance(value, basestring):
            value = rules[value]
        rules[key] = value
    for_each_rule(rules, follow_chains)

    def make_immutable(key, value): # Transform lists to tuples
        assert not isinstance(value, basestring) # Previous operation should have guaranteed this
        rules[key] = tuple(value)
    for_each_rule(rules, make_immutable)

    # Remove unreachable rules
    rules_old = rules
    rules = {}
    to_insert = ['start']
    while len(to_insert) > 0:
        key = to_insert[0]
        del to_insert[0]
        if not isinstance(key, basestring):
            if key[0] != 'token':
                to_insert.extend(key[num_leading_params[key[0]]:])
        elif key not in rules:
            value = rules_old[key]
            rules[key] = value
            assert not isinstance(value, basestring) # Previous operation should have guaranteed this
            if value[0] != 'token':
                for i in xrange(num_leading_params[value[0]], len(value)):
                    to_insert.append(value[i])

    return rules


grammar_rules = normalize_rules(grammar_rules)


class RecursionException(Exception):
    pass


def recursion_safe_cached(f):
    cache = {}
    computations_in_progress = set()
    @functools.wraps(f)
    def wrapped(key):
        if key not in cache:
            if key in computations_in_progress:
                raise RecursionException
            try:
                computations_in_progress.add(key)
                cache[key] = f(key)
            finally:
                computations_in_progress.remove(key)
        return cache[key]
    return wrapped


def semantic_elem_to_tokens(semantic_elem):
    if isinstance(semantic_elem, tokenize.Token):
        return [semantic_elem]
    else:
        assert isinstance(semantic_elem, parse.TreeNode)
        result = []
        for child_node in semantic_elem.child_nodes:
            result.extend(semantic_elem_to_tokens(child_node))
        return result


@recursion_safe_cached
def min_expansion(key):
    """Find the smallest possible semantic element that satisfies the
    given grammar rule, as counted by number of tokens.
    """
    rule = grammar_rules[key]
    if rule[0] == 'token':
        return rule[1]
    if rule[0] == 'either':
        result = None
        result_len = None
        for alt in rule[1:]:
            try:
                candidate = min_expansion(alt)
                candidate_len = len(semantic_elem_to_tokens(candidate))
                if result is None or candidate_len < result_len:
                    result = candidate
                    result_len = candidate_len
            except RecursionException:
                pass
        if result is None:
            raise RecursionException
        return result
    assert rule[0] == 'sequence'
    type = rule[1]
    child_expansions = []
    for child_rule in rule[3:]:
        child_expansion = min_expansion(child_rule)
        child_expansions.append(child_expansion)
    return parse.TreeNode(type, *child_expansions)


def find_rule_enclosers():
    """Find an encloser for each grammar rule.  An encloser is a function
    which, given a semantic element that satisfies the rule, encloses
    it in other semantic elements to form a semantic element that
    satisfies the start rule.
    """
    enclosers = {'start': lambda x: x}
    to_search = ['start']

    def add_encloser(key, f):
        if key in enclosers:
            return
        enclosers[key] = f
        to_search.append(key)

    def make_sequence_encloser(f, type, lhs, rhs):
        return lambda x: f(parse.TreeNode(type, *tuple(lhs + [x] + rhs)))

    while len(to_search) > 0:
        key = to_search[0]
        del to_search[0]
        f = enclosers[key]
        value = grammar_rules[key]
        if value[0] == 'either':
            for i in xrange(num_leading_params[value[0]], len(value)):
                add_encloser(value[i], f)
        elif value[0] == 'sequence':
            type = value[1]
            params = value[num_leading_params[value[0]]:]
            expansions = [min_expansion(x) for x in params]
            for i in xrange(len(params)):
                add_encloser(params[i], make_sequence_encloser(f, type, expansions[0:i], expansions[i+1:]))

    return enclosers


rule_enclosers = find_rule_enclosers()


def pretty_print(x):
    if isinstance(x, tokenize.Token) or isinstance(x, parse.TreeNode):
        x = semantic_elem_to_tokens(x)
    return ' '.join([t.str for t in x])


def make_test_elems():
    test_elems = []
    for key in grammar_rules:
        value = grammar_rules[key]
        if value[0] == 'sequence':
            type = value[1]
            params = value[num_leading_params[value[0]]:]
            expansions = [min_expansion(x) for x in params]
            for i in xrange(len(params)):
                lhs = expansions[0:i]
                rhs = expansions[i+1:]
                sub_rule = grammar_rules[params[i]]
                if sub_rule[0] == 'either':
                    for alt in sub_rule[num_leading_params[sub_rule[0]]:]:
                        elem = rule_enclosers[key](parse.TreeNode(type, *tuple(lhs + [min_expansion(alt)] + rhs)))
                        test_elems.append(elem)
        else:
            test_elems.append(rule_enclosers[key](min_expansion(key)))
    return test_elems


test_elems = make_test_elems()


def semantically_equal(e1, e2):
    if isinstance(e1, parse.TreeNode):
        if not isinstance(e2, parse.TreeNode):
            return False
        if e1.type != e2.type:
            return False
        if len(e1.child_nodes) != len(e2.child_nodes):
            return False
        for i in xrange(len(e1.child_nodes)):
            if not semantically_equal(e1.child_nodes[i], e2.child_nodes[i]):
                return False
        return True
    else:
        assert isinstance(e1, tokenize.Token)
        if not isinstance(e2, tokenize.Token):
            return False
        if e1.type != e2.type:
            return False
        if e1.str != e2.str:
            return False
        return True


class TestParse(unittest.TestCase):
    def test_basic(self):
        num_tests = 0
        random.seed(0)
        for r, semantic_elem in sorted([(random.uniform(0,1), elem) for elem in test_elems]):
            num_tests += 1
            tokens = semantic_elem_to_tokens(semantic_elem)
            #print pretty_print(tokens)
            parser = parse.Parser(tokens)
            try:
                parser.top()
                result = parser.result
                self.assertTrue(semantically_equal(semantic_elem, result))
            except Exception:
                if len(sys.exc_info()[1].args) <= 1:
                    msg_prefix = 'on expression number %r\n' % num_tests
                    msg_prefix += 'while parsing %s\n' % ' '.join([t.str for t in tokens])
                    msg_prefix += 'semantic_elem: %s\n' % parse.parse_tree_to_string(semantic_elem)
                    if parser.partial_result is not None:
                        msg_prefix += '       result: %s\n' % parse.parse_tree_to_string(parser.partial_result)
                    if len(sys.exc_info()[1].args) == 1:
                        sys.exc_info()[1].args = (msg_prefix + sys.exc_info()[1].args[0],)
                    else:
                        sys.exc_info()[1].args = (msg_prefix,)
                raise
        print 'Tested %r expressions' % num_tests


if __name__ == '__main__':
    unittest.main()
