"""Download the Tiny Shakespeare dataset (~1 MB) to data/input.txt.

    python data/download.py

Source: Andrej Karpathy's char-rnn Tiny Shakespeare corpus.
"""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    out = Path(__file__).parent / "input.txt"
    if out.exists():
        print(f"already present: {out} ({out.stat().st_size:,} bytes)")
        return
    print(f"downloading Tiny Shakespeare -> {out}")
    urllib.request.urlretrieve(URL, out)
    print(f"done: {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
