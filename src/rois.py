from typing import List

from src.constants import (
    ID_SERIES,
    FIRST_NAME,
    LAST_NAME,
    SEX,
    BIRTHDAY,
    ID_EMITTED_AT,
    ID_EXPIRES_AT,
    ID_EMITTED_BY,
)

# regions of interest from the face of ID
from src.types import Point, Roi

ROIS_FACE: List[Roi] = [
    Roi(ID_SERIES, top_left=Point(982, 45), bottom_right=Point(1264, 102)),
    Roi(FIRST_NAME, top_left=Point(460, 221), bottom_right=Point(1218, 274)),
    Roi(LAST_NAME, top_left=Point(463, 305), bottom_right=Point(1171, 354)),
    Roi(SEX, top_left=Point(843, 384), bottom_right=Point(1019, 434), whitelist="MF"),
    Roi(BIRTHDAY, top_left=Point(468, 464), bottom_right=Point(853, 515)),
    Roi(ID_EMITTED_AT, top_left=Point(468, 543), bottom_right=Point(853, 595)),
    Roi(ID_EXPIRES_AT, top_left=Point(468, 623), bottom_right=Point(853, 675)),
    Roi(ID_EMITTED_BY, top_left=Point(557, 713), bottom_right=Point(1243, 770)),
]
