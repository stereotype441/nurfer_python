import unittest

class PushBackIter(object):
    """Iterator that can be "backed up" by pushing items back onto it."""
    def __init__(self, iterable):
        self._iter = iterable.__iter__()
        self._push_back = []
        self._done = False

    def __iter__(self):
        return self

    def next(self):
        if len(self._push_back) != 0:
            x = self._push_back[-1]
            del self._push_back[-1]
            return x
        elif self._done:
            raise StopIteration
        else:
            try:
                return self._iter.next()
            except StopIteration:
                self._done = True
                raise

    def push_back(self, x):
        """Push a value back, so that the next call to next() will return it
        instead of advancing the iterator.
        """
        self._push_back.append(x)

    def peek(self):
        """Inspect the next value but don't pop it.  Return None if at end."""
        try:
            x = self.next()
        except StopIteration:
            return None
        self.push_back(x)
        return x


class TestPushBackIter(unittest.TestCase):
    def test_normal(self):
        collection = ['a', 'b', 'c']
        result = [x for x in PushBackIter(collection)]
        self.assertEqual(collection, result)

    def test_push_back(self):
        collection = ['a', 'b', 'c']
        iter = PushBackIter(collection)
        self.assertEqual('a', iter.peek())
        self.assertEqual('a', iter.next())
        iter.push_back('a1')
        self.assertEqual('a1', iter.next())
        self.assertEqual('b', iter.next())
        iter.push_back('b1')
        iter.push_back('a2')
        self.assertEqual('a2', iter.next())
        self.assertEqual('b1', iter.next())
        self.assertEqual('c', iter.next())
        self.assertEqual(None, iter.peek())
        iter.push_back('c1')
        self.assertEqual('c1', iter.next())
        self.assertRaises(StopIteration, iter.next)
        iter.push_back('c2')
        self.assertEqual('c2', iter.next())
        self.assertRaises(StopIteration, iter.next)


if __name__ == '__main__':
    unittest.main()
