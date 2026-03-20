from pathlib import Path
import sys


MODULE_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT_STR = str(MODULE_ROOT)

if MODULE_ROOT_STR not in sys.path:
    sys.path.insert(0, MODULE_ROOT_STR)
