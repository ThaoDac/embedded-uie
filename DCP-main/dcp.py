"""
Dark Channel Prior (DCP) Implementation
========================================
Based on the paper:
"Single Image Haze Removal Using Dark Channel Prior"
by Kaiming He, Jian Sun, Xiaoou Tang
CVPR 2009 / TPAMI 2011

This implementation provides image dehazing using the Dark Channel Prior method.
"""

import cv2
import math
import numpy as np


class DCPProcessor:
    """
    Dark Channel Prior processor for image dehazing.
    """

    def __init__(self, patch_size=15, omega=0.95, t0=0.1, guided_r=60, guided_eps=0.0001):
        """
        Initialize DCP processor.

        Args:
            patch_size: Local patch size for dark channel (default: 15)
            omega: Haze retention parameter (default: 0.95)
            t0: Lower bound for transmission to avoid noise (default: 0.1)
            guided_r: Radius for guided filter (default: 60)
            guided_eps: Regularization for guided filter (default: 0.0001)
        """
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.guided_r = guided_r
        self.guided_eps = guided_eps

    def dark_channel(self, image, patch_size=None):
        """
        Calculate Dark Channel using minimum across all RGB channels.

        Dark Channel Prior: In non-sky patches, at least one color channel
        has very low intensity at some pixels.

        Args:
            image: Input image (H, W, 3) in BGR format, normalized to [0, 1]
            patch_size: Size of local patch (default: use self.patch_size)

        Returns:
            dark_channel: Dark channel map (H, W), normalized to [0, 1]
        """
        if patch_size is None:
            patch_size = self.patch_size

        print(f"[DCP] Calculating dark channel with patch size {patch_size}...")

        # Split BGR channels
        b, g, r = cv2.split(image)

        # Take minimum across all three color channels
        dc = cv2.min(cv2.min(r, g), b)

        # Create rectangular structuring element for morphological erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

        # Apply erosion to get dark channel (minimum filter)
        dark = cv2.erode(dc, kernel)

        print(f"[DCP] Dark channel - Min: {dark.min():.4f}, Max: {dark.max():.4f}, Mean: {dark.mean():.4f}")

        return dark

    def atmospheric_light(self, image, dark_channel):
        """
        Estimate atmospheric light from the brightest pixels in dark channel.

        Algorithm:
        1. Pick the top 0.1% brightest pixels in the dark channel
        2. Among these pixels, select the one with highest intensity in the input image

        Args:
            image: Input image (H, W, 3) in BGR format, normalized to [0, 1]
            dark_channel: Dark channel map (H, W), normalized to [0, 1]

        Returns:
            A: Atmospheric light [B, G, R], shape (1, 3), normalized to [0, 1]
        """
        print(f"[DCP] Estimating atmospheric light from top 0.1% brightest pixels...")

        # Get image dimensions
        h, w = image.shape[:2]
        imsz = h * w

        # Number of pixels to consider (top 0.1%)
        numpx = int(max(math.floor(imsz / 1000), 1))

        # Flatten arrays
        darkvec = dark_channel.reshape(imsz)
        imvec = image.reshape(imsz, 3)

        # Get indices of brightest pixels in dark channel
        indices = darkvec.argsort()
        indices = indices[imsz - numpx::]

        # Calculate atmospheric light as average of brightest pixels
        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx

        print(f"[DCP] Atmospheric Light: B={A[0,0]:.4f}, G={A[0,1]:.4f}, R={A[0,2]:.4f}")

        return A

    def transmission_estimate(self, image, A, patch_size=None, omega=None):
        """
        Estimate transmission map using DCP.

        Formula: t(x) = 1 - omega * dark_channel(I(x) / A)

        Args:
            image: Input image (H, W, 3) in BGR format, normalized to [0, 1]
            A: Atmospheric light (1, 3), normalized to [0, 1]
            patch_size: Patch size for dark channel (default: use self.patch_size)
            omega: Haze retention parameter (default: use self.omega)

        Returns:
            transmission: Transmission map (H, W), normalized to [0, 1]
        """
        if patch_size is None:
            patch_size = self.patch_size
        if omega is None:
            omega = self.omega

        print(f"[DCP] Estimating transmission map with omega={omega}...")

        # Normalize image by atmospheric light
        im3 = np.empty(image.shape, image.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = image[:, :, ind] / A[0, ind]

        # Calculate dark channel of normalized image
        dark_norm = self.dark_channel(im3, patch_size)

        # Estimate transmission: t(x) = 1 - omega * dark_channel(I/A)
        transmission = 1 - omega * dark_norm

        print(f"[DCP] Transmission (raw) - Min: {transmission.min():.4f}, Max: {transmission.max():.4f}, Mean: {transmission.mean():.4f}")

        return transmission

    def guided_filter(self, I, p, r=None, eps=None):
        """
        Apply guided filter to refine transmission map.
        This preserves edges while smoothing the transmission map.

        Reference: "Guided Image Filtering" by Kaiming He et al., ECCV 2010

        Args:
            I: Guidance image (H, W), grayscale, normalized to [0, 1]
            p: Input image to be filtered (H, W), normalized to [0, 1]
            r: Radius of local window (default: use self.guided_r)
            eps: Regularization parameter (default: use self.guided_eps)

        Returns:
            q: Filtered output (H, W), normalized to [0, 1]
        """
        if r is None:
            r = self.guided_r
        if eps is None:
            eps = self.guided_eps

        print(f"[DCP] Applying guided filter (r={r}, eps={eps})...")

        # Calculate local mean
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        # Calculate local variance
        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        # Calculate linear coefficients
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Calculate output
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * I + mean_b

        print(f"[DCP] Transmission (refined) - Min: {q.min():.4f}, Max: {q.max():.4f}, Mean: {q.mean():.4f}")

        return q

    def transmission_refine(self, image, transmission):
        """
        Refine transmission map using guided filter.

        Args:
            image: Original image (H, W, 3) in BGR format, uint8 [0, 255]
            transmission: Estimated transmission map (H, W), normalized to [0, 1]

        Returns:
            t_refined: Refined transmission map (H, W), normalized to [0, 1]
        """
        print(f"[DCP] Refining transmission map...")

        # Convert image to grayscale for guidance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255.0

        # Apply guided filter
        t_refined = self.guided_filter(gray, transmission)

        return t_refined

    def recover(self, image, transmission, A, t0=None):
        """
        Recover scene radiance using the atmospheric scattering model.

        Formula: J(x) = (I(x) - A) / max(t(x), t0) + A

        This is the Inverse Foggy Model (IFM).

        Args:
            image: Input image (H, W, 3) in BGR format, normalized to [0, 1]
            transmission: Transmission map (H, W), normalized to [0, 1]
            A: Atmospheric light (1, 3), normalized to [0, 1]
            t0: Lower bound for transmission (default: use self.t0)

        Returns:
            recovered: Recovered image (H, W, 3), normalized to [0, 1]
        """
        if t0 is None:
            t0 = self.t0

        print(f"[DCP] Recovering scene using IFM (t0={t0})...")

        # Initialize result
        res = np.empty(image.shape, image.dtype)

        # Clip transmission to avoid division by very small numbers
        t = cv2.max(transmission, t0)

        # Recover scene radiance for each channel
        for ind in range(0, 3):
            res[:, :, ind] = (image[:, :, ind] - A[0, ind]) / t + A[0, ind]

        # Clip to valid range [0, 1]
        res = np.clip(res, 0, 1)

        print(f"[DCP] Scene recovery completed!")

        return res

    def process(self, img):
        """
        Complete DCP processing pipeline.

        Args:
            img: Input hazy image (H, W, 3) in BGR format, uint8 [0, 255]

        Returns:
            dict containing:
                - dark_channel: Dark channel map, uint8 [0, 255]
                - transmission_raw: Raw transmission map, uint8 [0, 255]
                - transmission_refined: Refined transmission map, uint8 [0, 255]
                - transmission_float: Refined transmission (float) for heatmap
                - atmospheric_light: Atmospheric light [B, G, R], uint8 [0, 255]
                - recovered_image: Dehazed image, uint8 [0, 255]
                - trans_stats: Statistics of transmission map
        """
        print("\n" + "="*70)
        print("DCP PROCESSING PIPELINE")
        print("="*70)
        print(f"Input image shape: {img.shape}, dtype: {img.dtype}")

        # Normalize image to [0, 1] for processing
        I = img.astype('float64') / 255.0

        # Step 1: Calculate dark channel
        print("\n[Step 1/5] Dark Channel Calculation")
        dark = self.dark_channel(I, self.patch_size)

        # Step 2: Estimate atmospheric light
        print("\n[Step 2/5] Atmospheric Light Estimation")
        A = self.atmospheric_light(I, dark)

        # Step 3: Estimate transmission
        print("\n[Step 3/5] Transmission Map Estimation")
        te = self.transmission_estimate(I, A, self.patch_size)

        # Step 4: Refine transmission with guided filter
        print("\n[Step 4/5] Transmission Map Refinement")
        t = self.transmission_refine(img, te)

        # Step 5: Recover scene
        print("\n[Step 5/5] Scene Recovery (IFM)")
        J = self.recover(I, t, A, self.t0)

        # Convert back to uint8 [0, 255]
        dark_uint8 = (dark * 255).astype(np.uint8)
        te_uint8 = (te * 255).astype(np.uint8)
        t_uint8 = (t * 255).astype(np.uint8)
        A_uint8 = (A * 255).astype(np.uint8)
        J_uint8 = (J * 255).astype(np.uint8)

        # Calculate transmission statistics
        trans_stats = {
            'min': float(t.min()),
            'max': float(t.max()),
            'mean': float(t.mean())
        }

        print("\n" + "="*70)
        print("DCP PROCESSING COMPLETED!")
        print("="*70 + "\n")

        return {
            'dark_channel': dark_uint8,
            'transmission_raw': te_uint8,
            'transmission_refined': t_uint8,
            'transmission_float': t,  # Keep float version for heatmap
            'atmospheric_light': A_uint8,
            'recovered_image': J_uint8,
            'trans_stats': trans_stats
        }


if __name__ == '__main__':
    print("DCP Processor Module")
    print("This module implements the Dark Channel Prior algorithm for image dehazing.")
    print("For usage, please import this module in your application.")