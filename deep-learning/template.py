import sys
from pathlib import Path
from typing import override

# include same dir layer.
same_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, same_dir)



class clazz:
    """ One-line class description.

    Long class description.

    Attributes:
        public_x: xxx
        public_y

    Examples:
        y = x
    """

    @override
    def __init__(self, x: int, y: float) -> tuple[str, bool]:
        """ One-line function description

        Long function description.

        Args:
            arg1: xxx
            arg2: xxx

        Returns:
            (ret1, ret2): xxx
            ret3: xxx

        Raises:

        Examples:

        """

