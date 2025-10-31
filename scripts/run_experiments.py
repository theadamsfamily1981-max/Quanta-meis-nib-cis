#!/usr/bin/env python3

import sys
import subprocess

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))
