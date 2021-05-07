class ImageResult:
    def __init__(
            self,
            id: str,
            failed: bool = False,
            area: float = None,
            solidity: float = None,
            max_width: int = None,
            max_height: int = None,
            avg_curve: float = None,
            n_leaves: int = None):
        self.id = id
        self.failed = failed
        self.area = area
        self.solidity = solidity
        self.max_width = max_width
        self.max_height = max_height
        self.avg_curve = avg_curve
        self.n_leaves = n_leaves