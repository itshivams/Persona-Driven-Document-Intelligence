import time, contextlib
from typing import Generator

@contextlib.contextmanager
def timed(label: str):
    start = time.time()
    yield
    dur = time.time() - start
    print(f"{label} finished in {dur:.2f}s")
