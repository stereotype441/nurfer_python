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
