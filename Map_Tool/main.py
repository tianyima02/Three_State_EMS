# /// script
# dependencies = [
#   "numpy",
#   "pygame-ce",
# ]
# ///

import asyncio

import numpy  # noqa: F401
import pygame  # noqa: F401

from map_hypercube import main_async


if __name__ == "__main__":
    asyncio.run(main_async())
