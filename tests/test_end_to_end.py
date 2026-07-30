import os
from subprocess import Popen, PIPE
import sys


def test_stdout_and_stderr_passthrough() -> None:
    """
    stdout and stderr are passed from the subprocess.
    """
    p = Popen(
        [
            sys.executable,
            "-m",
            "profila",
            "annotate",
            "--",
            "-c",
            "import sys; sys.stderr.write('err1@@\\nXX\\n'); sys.stdout.write('out2@@\\nYY\\n')",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    assert p.stdout is not None
    assert p.stderr is not None
    err_output = p.stderr.read()
    out_output = p.stdout.read()
    assert b"err1@@\nXX" in err_output
    assert b"out2@@\nYY" in out_output
