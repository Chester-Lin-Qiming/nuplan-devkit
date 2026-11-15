from __future__ import annotations

import os
from functools import cached_property
from typing import BinaryIO, Optional, Union, cast

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image as PilImage
from turbojpeg import TJPF_RGB, TurboJPEG

_JPEG_DECODER: Optional[TurboJPEG] = None
_JPEG_DECODER_PID: Optional[int] = None


def get_jpeg_decoder() -> TurboJPEG:
    """
    Process-local libjpeg-turbo decoder.
    `TurboJPEG()` dlopens the library and calls tjInitDecompress(), so it is only created once per
    process; the handle is keyed on the pid because DataLoader workers are forked and a ctypes handle
    must not be inherited across a fork.
    :return: The TurboJPEG instance for the current process.
    """
    global _JPEG_DECODER, _JPEG_DECODER_PID
    pid = os.getpid()
    if _JPEG_DECODER is None or _JPEG_DECODER_PID != pid:
        _JPEG_DECODER = TurboJPEG()
        _JPEG_DECODER_PID = pid
    return _JPEG_DECODER


class Image:
    """
    A class to represent an image. This class is an analogue to LidarPointCloud. It is a class for manipulating and
    transforming an image. Any transformation functions (flip, scale, translate) should be added to this class in the
    future.
    """

    def __init__(self, image: Union[PilImage.Image, npt.NDArray[np.uint8]]) -> None:
        """
        Constructor for the Image class.
        :param image: A PIL.Image.Image, or the uint8 RGB array produced by libjpeg-turbo.
        """
        self._image = image

    @property
    def as_pil(self) -> PilImage.Image:
        """
        Returns the image of type PIL.Image.Image in uint8, RGB format.
        :return: An image of type PIL.Image.Image.
        """
        if isinstance(self._image, np.ndarray):
            self._image = PilImage.fromarray(self._image)
        return self._image

    @cached_property
    def as_numpy(self) -> npt.NDArray[np.uint8]:
        """
        Returns the image as a numpy array in uint8, RGB format.
        :return: An image as a numpy array.
        """
        return self.as_numpy_nocache()

    def as_numpy_nocache(self) -> npt.NDArray[np.uint8]:
        """
        Returns the image as a numpy array in uint8, RGB format. A non caching variation to save on memory if needed.
        :return: An image as a numpy array.
        """
        if isinstance(self._image, np.ndarray):
            # Already decoded by libjpeg-turbo; hand it out without a copy.
            return self._image
        return np.array(self._image, dtype=np.uint8)

    @cached_property
    def as_cv2(self) -> npt.NDArray[np.uint8]:
        """
        Returns the image as a CV2 image in uint8, BGR format. It is a numpy array under the hood.
        This function is a convenience for to be used with cv2.imshow().
        :return: An image as a CV2 image.
        """
        return self.as_cv2_nocache()

    def as_cv2_nocache(self) -> npt.NDArray[np.uint8]:
        """
        Returns the image as a CV2 image in uint8, BGR format. It is a numpy array under the hood. This function
        is a convenience for to be used with cv2.imshow(). A non caching variation to save on memory if needed.
        :return: An image as a CV2 image.
        """
        return cast(npt.NDArray[np.uint8], cv2.cvtColor(self.as_numpy_nocache(), cv2.COLOR_RGB2BGR))

    @classmethod
    def from_buffer(cls, blob: BinaryIO) -> Image:
        """
        Instantiates Image from buffer.

        The JPEG is decoded with libjpeg-turbo instead of Pillow (~2.5x faster on the nuPlan camera
        frames, pixel-identical). TJPF_RGB is requested explicitly because `decode` defaults to
        TJPF_BGR, which would silently swap the red and blue channels relative to Pillow's output.
        :param blob: Data to load.
        :return: An Image object.
        """
        return cls(get_jpeg_decoder().decode(blob.read(), pixel_format=TJPF_RGB))
