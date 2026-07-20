"""Console-script entry points for edge-vision.

The packaged ``edgevision-smoke`` command (declared in ``pyproject.toml`` under
``[project.scripts]``) resolves to :func:`edgevision.cli.smoke.main`. Keeping
the entry point inside the installed ``edgevision`` package — rather than the
top-level ``scripts/`` directory, which is *not* part of the ``src`` packages
root and therefore not importable after a clean ``pip install -e .`` — means the
advertised command runs from any working directory.
"""
