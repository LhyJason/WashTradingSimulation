"""detection -- Part 2: detecting wash trading in the simulated market (capstone).

Container package; concrete detector versions live in subpackages (v1, ...), mirroring
``washtrade``. Detectors read only a run's ``observable/`` folder (see
``washtrade/v3/export.py``); ``labels/`` is touched only by the evaluation code.

Not importable as ``capstone codes.detection`` (space in the folder name); entry points add
``<repo>/capstone codes`` to sys.path and then ``import detection.v1``.
"""
