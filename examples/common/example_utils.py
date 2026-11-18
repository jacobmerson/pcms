"""Helpers shared by the Python examples.

The Python counterpart of example_utils.hpp: it tallies the checks an example
asserts, so that every example prints what it verified and exits with a
nonzero status if a check failed. Import it from an example with::

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                           / "common"))
    from example_utils import Report
"""


class Report:
    """Tallies the checks of an example."""

    def __init__(self):
        self.checks = 0
        self.failures = 0

    def check_at_most(self, what, error, tolerance):
        """Check that `error` (a norm, a size) is at most `tolerance`."""
        self._record(what, error <= tolerance,
                     "%.3e <= %.1e" % (error, tolerance))

    def check(self, what, holds, value):
        """Check a statement, with the number it is about."""
        self._record(what, holds, "%d" % value)

    def finish(self):
        """Print the verdict; return the exit code of the program."""
        if self.failures == 0:
            print("SUCCESS: %d checks passed" % self.checks)
            return 0
        print("FAILED: %d of %d checks failed" % (self.failures, self.checks))
        return 1

    def _record(self, what, ok, detail):
        print("  [%s] %s: %s" % ("ok" if ok else "FAIL", what, detail))
        self.checks += 1
        self.failures += 0 if ok else 1
