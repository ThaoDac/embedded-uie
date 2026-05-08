"""
ULAP - Underwater Light Attenuation Prior
==========================================
This module implements the ULAP (Underwater Light Attenuation Prior) algorithm
for underwater image enhancement.

Pipeline:
1. Depth Map Estimation
2. Global Histogram Stretching
3. Background Light Estimation
4. Minimum Depth Calculation
5. RGB Transmission Estimation
6. Transmission Map Refinement (Guided Filter)
7. Scene Radiance Recovery (IFM - Image Formation Model)
"""

import numpy as np
import cv2
import math


# ============================================================================
# Guided Filter Implementation
# ============================================================================

class GuidedFilter:
    """
    Guided Filter for edge-preserving smoothing.
    Used for refining transmission maps.
    """

    def __init__(self, I, radius=5, epsilon=0.4):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = self._toFloatImg(I)
        self._initFilter()

    def _toFloatImg(self, img):
        if img.dtype == np.float32:
            return img
        return (1.0 / 255.0) * np.float32(img)

    def _initFilter(self):
        I = self._I
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir ** 2, (r, r)) - self._Ir_mean ** 2 + eps
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps

        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        Irr_inv /= I_cov
        Irg_inv /= I_cov
        Irb_inv /= I_cov
        Igg_inv /= I_cov
        Igb_inv /= I_cov
        Ibb_inv /= I_cov

        self._Irr_inv = Irr_inv
        self._Irg_inv = Irg_inv
        self._Irb_inv = Irb_inv
        self._Igg_inv = Igg_inv
        self._Igb_inv = Igb_inv
        self._Ibb_inv = Ibb_inv

    def _computeCoefficients(self, p):
        r = self._radius
        I = self._I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        p_mean = cv2.blur(p, (r, r))
        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean

        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov

        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        ar_mean = cv2.blur(ar, (r, r))
        ag_mean = cv2.blur(ag, (r, r))
        ab_mean = cv2.blur(ab, (r, r))
        b_mean = cv2.blur(b, (r, r))

        return ar_mean, ag_mean, ab_mean, b_mean

    def _computeOutput(self, ab, I):
        ar_mean, ag_mean, ab_mean, b_mean = ab
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        q = ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean
        return q

    def filter(self, p):
        p_32F = self._toFloatImg(p)
        ab = self._computeCoefficients(p)
        return self._computeOutput(ab, self._I)


# ============================================================================
# Step 1: Depth Map Estimation
# ============================================================================

def depthMap(img):
    """
    Estimate depth map from underwater image using learned coefficients.

    Args:
        img: Input BGR image (0-255)

    Returns:
        depth_map: Estimated depth map (unnormalized)
    """
    print("[ULAP] Step 1: Depth Map Estimation")

    theta_0 = 0.51157954
    theta_1 = 0.50516165
    theta_2 = -0.90511117

    img = img / 255.0
    x_1 = np.maximum(img[:, :, 0], img[:, :, 1])  # max(B, G)
    x_2 = img[:, :, 2]  # R channel

    depth_map = theta_0 + theta_1 * x_1 + theta_2 * x_2

    print(f"   ✓ Depth map estimated: shape={depth_map.shape}, range=[{depth_map.min():.4f}, {depth_map.max():.4f}]")

    return depth_map


# ============================================================================
# Step 2: Global Histogram Stretching
# ============================================================================

def global_stretching(img_L):
    """
    Apply global histogram stretching to normalize depth map.

    Args:
        img_L: Input depth map

    Returns:
        stretched: Normalized depth map (0-1)
    """
    print("[ULAP] Step 2: Global Histogram Stretching")

    height = len(img_L)
    width = len(img_L[0])
    length = height * width

    R_array = []
    for i in range(height):
        for j in range(width):
            R_array.append(img_L[i][j])

    R_array.sort()
    I_min = R_array[int(length / 2000)]
    I_max = R_array[-int(length / 2000)]

    print(f"   ✓ Stretching range: I_min={I_min:.4f}, I_max={I_max:.4f}")

    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                array_Global_histogram_stretching_L[i][j] = 0
            elif img_L[i][j] > I_max:
                array_Global_histogram_stretching_L[i][j] = 1
            else:
                p_out = (img_L[i][j] - I_min) * ((1 - 0) / (I_max - I_min)) + 0
                array_Global_histogram_stretching_L[i][j] = p_out

    return array_Global_histogram_stretching_L


# ============================================================================
# Step 3: Background Light Estimation
# ============================================================================

def BLEstimation(img, DepthMap):
    """
    Estimate background light (atmospheric light) from brightest pixels.

    Args:
        img: Input BGR image (0-255 or 0-1)
        DepthMap: Estimated depth map

    Returns:
        A: Background light RGB values (0-1)
    """
    print("[ULAP] Step 3: Background Light Estimation")

    h, w, c = img.shape
    if img.dtype == np.uint8:
        img = np.float32(img) / 255

    n_bright = int(np.ceil(0.001 * h * w))
    reshaped_Jdark = DepthMap.reshape(1, -1)
    Y = np.sort(reshaped_Jdark)
    Loc = np.argsort(reshaped_Jdark)
    Ics = img.reshape(1, h * w, 3)
    ix = img.copy()
    dx = DepthMap.reshape(1, -1)

    Acand = np.zeros((1, n_bright, 3), dtype=np.float32)
    Amag = np.zeros((1, n_bright, 1), dtype=np.float32)

    for i in range(n_bright):
        x = Loc[0, h * w - 1 - i]
        j = int(x / w)
        k = int(x % w)
        ix[j, k, 0] = 0
        ix[j, k, 1] = 0
        ix[j, k, 2] = 1
        Acand[0, i, :] = Ics[0, Loc[0, h * w - 1 - i], :]
        Amag[0, i] = np.linalg.norm(Acand[0, i, :])

    reshaped_Amag = Amag.reshape(1, -1)
    Y2 = np.sort(reshaped_Amag)
    Loc2 = np.argsort(reshaped_Amag)
    A_1 = Acand[0, Loc2[0, (n_bright - 1):n_bright], :]
    A_1 = A_1[0]

    print(f"   ✓ Background light (A): B={A_1[0]:.4f}, G={A_1[1]:.4f}, R={A_1[2]:.4f}")

    return A_1


# ============================================================================
# Step 4: Minimum Depth Calculation
# ============================================================================

def minDepth(img, BL):
    """
    Calculate minimum depth based on background light.

    Args:
        img: Input BGR image (0-255)
        BL: Background light (0-255)

    Returns:
        min_depth: Minimum depth value
    """
    print("[ULAP] Step 4: Minimum Depth Calculation")

    img = img / 255.0
    BL = BL / 255.0
    Max = []
    img = np.float32(img)

    for i in range(0, 3):
        Max_Abs = np.absolute(img[i] - BL[i])
        Max_I = np.max(Max_Abs)
        Max_B = np.max([BL[i], (1 - BL[i])])
        temp = Max_I / Max_B
        Max.append(temp)

    K_b = np.max(Max)
    min_depth = 1 - K_b

    print(f"   ✓ Minimum depth (d_0): {min_depth:.4f}")

    return min_depth


# ============================================================================
# Step 5: RGB Transmission Estimation
# ============================================================================

def getRGBTransmissionESt(depth_map):
    """
    Estimate RGB transmission maps from depth map using exponential decay.

    Args:
        depth_map: Depth map

    Returns:
        transmissionB, transmissionG, transmissionR: Transmission maps for each channel
    """
    print("[ULAP] Step 5: RGB Transmission Estimation")

    # Different attenuation coefficients for each channel
    # Blue light penetrates deepest, red light is attenuated most
    transmissionB = 0.97 ** depth_map
    transmissionG = 0.95 ** depth_map
    transmissionR = 0.83 ** depth_map

    print(f"   ✓ Transmission maps estimated")
    print(f"      - transmissionR: range=[{transmissionR.min():.4f}, {transmissionR.max():.4f}]")
    print(f"      - transmissionG: range=[{transmissionG.min():.4f}, {transmissionG.max():.4f}]")
    print(f"      - transmissionB: range=[{transmissionB.min():.4f}, {transmissionB.max():.4f}]")

    return transmissionB, transmissionG, transmissionR


# ============================================================================
# Step 6: Refined Transmission Map (Guided Filter)
# ============================================================================

def refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img):
    """
    Refine transmission maps using Guided Filter for edge preservation.

    Args:
        transmissionB, transmissionG, transmissionR: Raw transmission maps
        img: Original image for guidance

    Returns:
        transmission: Refined 3-channel transmission map
    """
    print("[ULAP] Step 6: Transmission Map Refinement with Guided Filter")

    gimfiltR = 50  # Guided filter radius
    eps = 10 ** -3  # Epsilon value

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmissionB = guided_filter.filter(transmissionB)
    transmissionG = guided_filter.filter(transmissionG)
    transmissionR = guided_filter.filter(transmissionR)

    transmission = np.zeros(img.shape)
    transmission[:, :, 0] = transmissionB
    transmission[:, :, 1] = transmissionG
    transmission[:, :, 2] = transmissionR

    print(f"   ✓ Transmission maps refined with Guided Filter (radius={gimfiltR}, eps={eps})")

    return transmission


# ============================================================================
# Step 7: Scene Radiance Recovery (IFM)
# ============================================================================

def sceneRadianceRGB(img, transmission, AtomsphericLight):
    """
    Recover scene radiance using Image Formation Model (IFM).

    Formula: J(x) = (I(x) - A) / t(x) + A

    Args:
        img: Input image (0-255)
        transmission: Refined transmission map
        AtomsphericLight: Background light (0-1)

    Returns:
        sceneRadiance: Recovered image (0-255, uint8)
    """
    print("[ULAP] Step 7: Scene Radiance Recovery (IFM - Image Formation Model)")
    print("   Formula: J(x) = (I(x) - A) / t(x) + A")
    print(f"   Where:")
    print(f"      - I(x): Observed image (input)")
    print(f"      - J(x): Scene radiance (output - what we want)")
    print(f"      - t(x): Transmission map (light attenuation)")
    print(f"      - A: Background light = [B={AtomsphericLight[0]*255:.2f}, G={AtomsphericLight[1]*255:.2f}, R={AtomsphericLight[2]*255:.2f}]")

    sceneRadiance = np.zeros(img.shape)
    img = np.float16(img)

    for i in range(0, 3):
        # Apply IFM formula: J = (I - A) / t + A
        sceneRadiance[:, :, i] = (img[:, :, i] - AtomsphericLight[i]) / transmission[:, :, i] + AtomsphericLight[i]

        # Clip values to valid range [0, 255]
        for j in range(0, sceneRadiance.shape[0]):
            for k in range(0, sceneRadiance.shape[1]):
                if sceneRadiance[j, k, i] > 255:
                    sceneRadiance[j, k, i] = 255
                if sceneRadiance[j, k, i] < 0:
                    sceneRadiance[j, k, i] = 0

    sceneRadiance = np.uint8(sceneRadiance)

    print(f"   ✓ Scene radiance recovered successfully")

    return sceneRadiance


# ============================================================================
# Main ULAP Processing Function
# ============================================================================

def process_ulap(img_bgr, blockSize=9, gimfiltR=50, eps=0.001):
    """
    Complete ULAP pipeline for underwater image enhancement.

    Args:
        img_bgr: Input BGR image (numpy array, 0-255)
        blockSize: Block size for dark channel (default: 9)
        gimfiltR: Guided filter radius (default: 50)
        eps: Guided filter epsilon (default: 0.001)

    Returns:
        dict: Dictionary containing all intermediate results and final output
            - output: Enhanced image (BGR, uint8)
            - depth_map: Estimated depth map
            - depth_map_refined: Refined depth map after guided filter
            - background_light: Background light RGB values
            - min_depth: Minimum depth value
            - transmission_raw: Raw transmission maps (B, G, R)
            - transmission_refined: Refined transmission map
    """
    print("\n" + "=" * 80)
    print("ULAP - Underwater Light Attenuation Prior")
    print("=" * 80)

    # Step 1: Depth Map Estimation
    DepthMap = depthMap(img_bgr)

    # Step 2: Global Histogram Stretching
    DepthMap = global_stretching(DepthMap)

    # Refine depth map with guided filter
    print("[ULAP] Refining Depth Map with Guided Filter")
    guided_filter = GuidedFilter(img_bgr, gimfiltR, eps)
    refineDR = guided_filter.filter(DepthMap)
    refineDR = np.clip(refineDR, 0, 1)
    print(f"   ✓ Depth map refined: range=[{refineDR.min():.4f}, {refineDR.max():.4f}]")

    # Step 3: Background Light Estimation
    AtomsphericLight = BLEstimation(img_bgr, DepthMap) * 255

    # Step 4: Minimum Depth Calculation
    d_0 = minDepth(img_bgr, AtomsphericLight)
    d_f = 8 * (DepthMap + d_0)

    # Step 5: RGB Transmission Estimation
    transmissionB, transmissionG, transmissionR = getRGBTransmissionESt(d_f)

    # Step 6: Transmission Map Refinement
    transmission = refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img_bgr)

    # Step 7: Scene Radiance Recovery (IFM)
    sceneRadiance = sceneRadianceRGB(img_bgr, transmission, AtomsphericLight)

    print("=" * 80)
    print("✅ ULAP Processing Complete")
    print("=" * 80 + "\n")

    return {
        'output': sceneRadiance,
        'depth_map': DepthMap,
        'depth_map_refined': refineDR,
        'background_light': AtomsphericLight,
        'min_depth': d_0,
        'transmission_raw': (transmissionB, transmissionG, transmissionR),
        'transmission_refined': transmission
    }