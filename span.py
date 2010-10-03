class Span(object):
    def __init__(self, full_text, start, end):
        self._full_text = full_text
        self._start = start
        self._end = end

    @property
    def full_text(self):
        return self._full_text

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def sub_text(self):
        return self._full_text[self._start : self._end]

    def cut(self, cut_pos):
        span1 = self.__new__(type(self))
        span1.__dict__.update(self.__dict__)
        span1._end = cut_pos
        span2 = self.__new__(type(self))
        span2.__dict__.update(self.__dict__)
        span2._start = cut_pos
        return span1, span2

    def __repr__(self):
        other_properties = self.__dict__.copy()
        del other_properties['_full_text']
        del other_properties['_start']
        del other_properties['_end']
        return '<%s %r (%r-%r) %r>' % (type(self).__name__, self.sub_text, self.start, self.end, other_properties)
