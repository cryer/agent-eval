"""Zero-dependency checker: exits non-zero if the fixture code is buggy.

Used by cases_coding.yaml via `command_exit` — plain asserts instead of pytest
so the assertion works in any environment with just a Python interpreter.
"""

from calc import add, multiply

assert add(2, 3) == 5, f"add(2, 3) = {add(2, 3)}, expected 5"
assert add(-1, 1) == 0, f"add(-1, 1) = {add(-1, 1)}, expected 0"
assert multiply(2, 3) == 6, f"multiply(2, 3) = {multiply(2, 3)}, expected 6"
print("all checks passed")
