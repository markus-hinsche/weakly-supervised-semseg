import random
from pprint import pprint
import os
from pathlib import Path

from config import IMAGE_DATA_DIR

random.seed(42)

if __name__ == "__main__":
    areas = os.listdir(Path(IMAGE_DATA_DIR))  # 33 areas

    N1 = random.sample(areas, 3)
    areas = [area for area in areas if area not in N1]
    assert len(areas) == 30

    N2 = random.sample(areas, 23)
    areas = [area for area in areas if area not in N2]

    assert len(areas) == 7
    N3 = areas

    pprint(N1)
    pprint(N2)
    pprint(N3)
