import sys

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: list[str]) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    :param command: A list of shell commands as strings.
    """
    msg = None
    try:
        # use the interpreter running the tests, not the first `python` on PATH
        sh.Command(sys.executable)(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg)
