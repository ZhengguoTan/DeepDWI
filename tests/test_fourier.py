import unittest

import torch
import torch.testing as ptt

from deepdwi import fourier

if __name__ == "__main__":
    unittest.main()


class TestFourier(unittest.TestCase):
    def test_fft(self):
        input = torch.tensor([0, 1, 0], dtype=torch.cfloat)
        ptt.assert_close(
            fourier.fft(input), torch.ones(3, dtype=torch.cfloat) / 3**0.5,
            atol=1e-5, rtol=1e-5
        )

        input = torch.tensor([1, 1, 1], dtype=torch.cfloat)
        ptt.assert_close(
            fourier.fft(input), torch.tensor([0, 3**0.5, 0], dtype=torch.cfloat),
            atol=1e-5, rtol=1e-5
        )

        input = torch.randn([4, 5, 6])
        ptt.assert_close(
            fourier.fft(input),
            torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(input), norm="ortho")),
            atol=1e-5, rtol=1e-5
        )

    def test_fft_dtype(self):
        for dtype in [torch.complex64, torch.complex128]:
            input = torch.tensor([0, 1, 0], dtype=dtype)
            output = fourier.fft(input)

            assert output.dtype == dtype

    def test_ifft(self):
        input = torch.tensor([0, 1, 0], dtype=torch.cfloat)
        ptt.assert_close(
            fourier.ifft(input), torch.ones(3, dtype=torch.cfloat) / 3**0.5,
            atol=1e-5, rtol=1e-5
        )

        input = torch.tensor([1, 1, 1], dtype=torch.cfloat)
        ptt.assert_close(
            fourier.ifft(input), torch.tensor([0, 3**0.5, 0], dtype=torch.cfloat)
        )

        input = torch.randn([4, 5, 6])
        ptt.assert_close(
            fourier.ifft(input),
            torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(input), norm="ortho"))
        )
