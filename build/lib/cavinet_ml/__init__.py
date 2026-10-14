"""CaviNet AI pipeline (shared by the training toolkit and the application worker)."""

import warnings

__version__ = "0.9.0"


def hide_library_notices() -> None:
    """Hide deprecation notices that third-party libraries print on import (MONAI uses the
    deprecated torch.jit.interface). They need no action and would only confuse users."""
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.interface.*", category=FutureWarning)
