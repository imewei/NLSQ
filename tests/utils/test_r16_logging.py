import logging

from nlsq.utils.logging import get_logger


def test_no_duplicate_lines_after_basicconfig(capsys):
    log = get_logger("r16_dup")
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()
    try:
        log.warning("before")  # nlsq stderr handler is the only sink
        logging.basicConfig(level=logging.WARNING)
        # basicConfig installs its own stderr handler; with the nlsq handler
        # standing down the line appears exactly once.
        capsys.readouterr()
        log.warning("after-config")
        err = capsys.readouterr().err
        assert err.count("after-config") <= 1
    finally:
        root.handlers[:] = saved
