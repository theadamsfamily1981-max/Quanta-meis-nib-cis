import sys, pathlib
parent = pathlib.Path(__file__).resolve().parents[1]
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))
