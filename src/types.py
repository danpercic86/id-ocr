from typing import Tuple


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __call__(self, *args, **kwargs) -> Tuple[int, int]:
        return self.x, self.y


class Roi:
    def __init__(
            self,
            region_name: str,
            *,
            top_left: Point,
            bottom_right: Point,
            whitelist: str = None,
            blacklist: str = None,
    ):
        self.top_left = top_left
        self.region_name = region_name
        self.bottom_right = bottom_right
        self.blacklist = blacklist
        self.whitelist = whitelist
