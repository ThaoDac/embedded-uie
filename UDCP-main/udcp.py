"""
UDCP (Underwater Dark Channel Prior) Implementation
====================================================
This module implements the Underwater Dark Channel Prior technique for underwater image enhancement.

Main differences between UDCP and DCP:
- UDCP: Uses only Green and Blue channels (min(G, B)) for dark channel - optimized for underwater images
- DCP: Uses all RGB channels (min(R, G, B)) for dark channel - designed for hazy/foggy images

References:
- Carlevaris-Bianco, Nicholas, Anush Mohan, and Ryan M. Eustice.
  "Initial results in underwater single image dehazing."
  OCEANS 2010 MTS/IEEE SEATTLE. IEEE, 2010.
"""

import numpy as np
import cv2


# ============================================================================
# Node class for storing pixel positions during sorting
# ============================================================================

class Node(object):
    """Data structure to store pixel position and value for sorting"""
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print(self.x, self.y, self.value)


# ============================================================================
# Dark Channel Calculation (UDCP version - uses only G and B channels)
# ============================================================================

def getMinChannel(img):
    """
    Get minimum channel for UDCP (using only Green and Blue channels).
    This is the key difference from DCP which uses all RGB channels.

    Args:
        img: Input image in RGB format (numpy array)

    Returns:
        imgGray: Grayscale image with minimum of G and B channels
    """
    print("[UDCP] Bước 1: Tính Min Channel (chỉ sử dụng Green và Blue)")
    imgGray = np.zeros((img.shape[0], img.shape[1]), 'float32')

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 255
            # IMPORTANT: Only loop through channels 1 and 2 (G and B), skip channel 0 (R)
            for k in range(1, 3):  # k=1 is Green, k=2 is Blue
                if img.item((i, j, k)) < localMin:
                    localMin = img.item((i, j, k))
            imgGray[i, j] = localMin

    print(f"   - Min Channel shape: {imgGray.shape}")
    print(f"   - Min Channel range: [{imgGray.min():.2f}, {imgGray.max():.2f}]")
    return imgGray


def getDarkChannel(img, blockSize):
    """
    Calculate the dark channel using minimum filter on local patches.
    This applies a minimum filter over local blockSize x blockSize neighborhoods.

    Args:
        img: Input RGB image
        blockSize: Size of local patch (e.g., 9 for 9x9 patches)

    Returns:
        imgDark: Dark channel image
    """
    print(f"[UDCP] Bước 2: Tính Dark Channel với block size = {blockSize}x{blockSize}")

    # First get the minimum channel (G and B only for UDCP)
    img = getMinChannel(img)

    # Apply minimum filter over local patches
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # Pad the image with maximum value (255) for boundary handling
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 255
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img

    # Calculate dark channel by finding minimum in each local patch
    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 255
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin

    print(f"   - Dark Channel shape: {imgDark.shape}")
    print(f"   - Dark Channel range: [{imgDark.min()}, {imgDark.max()}]")
    return imgDark


# ============================================================================
# Atmospheric Light Estimation
# ============================================================================

def getAtomsphericLight(darkChannel, img):
    """
    Estimate atmospheric light from the brightest pixels in the dark channel.

    Args:
        darkChannel: Dark channel image
        img: Original RGB image

    Returns:
        atomsphericLight: Estimated atmospheric light (RGB values)
    """
    print("[UDCP] Bước 3: Ước lượng Atmospheric Light")

    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []

    # Store all pixels with their dark channel values
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)

    # Sort by dark channel value (descending) and take the brightest pixel
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    # Get atmospheric light from the brightest pixel in dark channel
    atomsphericLight = img[nodes[0].x, nodes[0].y, :]

    print(f"   - Atmospheric Light (R, G, B): {atomsphericLight}")
    print(f"   - Vị trí pixel sáng nhất trong Dark Channel: ({nodes[0].x}, {nodes[0].y})")
    return atomsphericLight


# ============================================================================
# Transmission Map Estimation
# ============================================================================

def getMinChannelNormalized(img, AtomsphericLight):
    """
    Get normalized minimum channel (for transmission estimation).
    Normalize image by atmospheric light before taking minimum.

    Args:
        img: Input RGB image
        AtomsphericLight: Estimated atmospheric light

    Returns:
        imgGrayNormalization: Normalized minimum channel
    """
    imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 1
            # Only use G and B channels (channels 1 and 2)
            for k in range(1, 3):
                imgNormalization = img.item((i, j, k)) / AtomsphericLight[k]
                if imgNormalization < localMin:
                    localMin = imgNormalization
            imgGrayNormalization[i, j] = localMin

    return imgGrayNormalization


def getTransmission(img, AtomsphericLight, blockSize):
    """
    Estimate initial transmission map using UDCP.
    Transmission map indicates how much light reaches the camera.

    Args:
        img: Input RGB image
        AtomsphericLight: Estimated atmospheric light
        blockSize: Size of local patch

    Returns:
        transmission: Estimated transmission map (clipped to [0.1, 0.9])
    """
    print(f"[UDCP] Bước 4: Ước lượng Transmission Map ban đầu")

    # Normalize image by atmospheric light
    img = getMinChannelNormalized(img, AtomsphericLight)

    # Apply minimum filter over local patches
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # Pad with 1 (maximum normalized value)
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 1
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img

    # Calculate transmission map
    imgDark = np.zeros((img.shape[0], img.shape[1]))
    localMin = 1
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 1
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin

    # Transmission is complement of normalized dark channel
    transmission = 1 - imgDark

    # Clip transmission to avoid extreme values
    transmission = np.clip(transmission, 0.1, 0.9)

    print(f"   - Transmission Map shape: {transmission.shape}")
    print(f"   - Transmission range (raw): [{transmission.min():.3f}, {transmission.max():.3f}]")
    print(f"   - Transmission mean: {transmission.mean():.3f}")

    return transmission


# ============================================================================
# Guided Filter Implementation
# ============================================================================

class GuidedFilter:
    """
    Guided Filter for edge-preserving smoothing.
    Used to refine the transmission map while preserving edges.

    Reference:
    - He, Kaiming, Jian Sun, and Xiaoou Tang. "Guided image filtering."
      IEEE transactions on pattern analysis and machine intelligence (2013).
    """

    def __init__(self, I, radius, epsilon):
        """
        Initialize Guided Filter.

        Args:
            I: Guidance image (RGB)
            radius: Radius of local window
            epsilon: Regularization parameter (prevents division by zero)
        """
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = self._toFloatImg(I)
        self._initFilter()

    def _toFloatImg(self, img):
        """Convert image to float32 in range [0, 1]"""
        if img.dtype == np.float32:
            return img
        return (1.0 / 255.0) * np.float32(img)

    def _initFilter(self):
        """Precompute filter coefficients"""
        I = self._I
        r = self._radius
        eps = self._epsilon

        # Split into RGB channels
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        # Calculate mean of each channel
        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        # Calculate covariance matrix elements
        Irr_var = cv2.blur(Ir ** 2, (r, r)) - self._Ir_mean ** 2 + eps
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps

        # Calculate inverse of covariance matrix (3x3 for RGB)
        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        # Normalize by determinant
        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        Irr_inv /= I_cov
        Irg_inv /= I_cov
        Irb_inv /= I_cov
        Igg_inv /= I_cov
        Igb_inv /= I_cov
        Ibb_inv /= I_cov

        # Store inverse covariance matrix
        self._Irr_inv = Irr_inv
        self._Irg_inv = Irg_inv
        self._Irb_inv = Irb_inv
        self._Igg_inv = Igg_inv
        self._Igb_inv = Igb_inv
        self._Ibb_inv = Ibb_inv

    def _computeCoefficients(self, p):
        """
        Compute linear coefficients for guided filtering.

        Args:
            p: Input image to be filtered

        Returns:
            Tuple of (ar_mean, ag_mean, ab_mean, b_mean) coefficients
        """
        r = self._radius
        I = self._I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        # Calculate correlation between guidance image and input
        p_mean = cv2.blur(p, (r, r))
        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        # Calculate covariance
        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean

        # Calculate linear coefficients using inverse covariance
        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov

        # Calculate bias term
        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        # Average coefficients over local window
        ar_mean = cv2.blur(ar, (r, r))
        ag_mean = cv2.blur(ag, (r, r))
        ab_mean = cv2.blur(ab, (r, r))
        b_mean = cv2.blur(b, (r, r))

        return ar_mean, ag_mean, ab_mean, b_mean

    def _computeOutput(self, ab, I):
        """
        Compute filtered output using linear model.

        Args:
            ab: Tuple of coefficients
            I: Guidance image

        Returns:
            q: Filtered output
        """
        ar_mean, ag_mean, ab_mean, b_mean = ab
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        q = ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean
        return q

    def filter(self, p):
        """
        Apply guided filter to input image.

        Args:
            p: Input image to be filtered

        Returns:
            Filtered image
        """
        p_32F = self._toFloatImg(p)
        ab = self._computeCoefficients(p)
        return self._computeOutput(ab, self._I)


def Refinedtransmission(transmission, img):
    """
    Refine transmission map using Guided Filter.
    This preserves edges while smoothing the transmission map.

    Args:
        transmission: Initial transmission map
        img: Original RGB image (used as guidance)

    Returns:
        transmission: Refined transmission map
    """
    print("[UDCP] Bước 5: Tinh chỉnh Transmission Map bằng Guided Filter")

    gimfiltR = 50  # Guided filter radius
    eps = 10 ** -3  # Epsilon for regularization

    print(f"   - Guided Filter parameters: radius={gimfiltR}, epsilon={eps}")

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmission = guided_filter.filter(transmission)
    transmission = np.clip(transmission, 0.1, 0.9)

    print(f"   - Refined Transmission range: [{transmission.min():.3f}, {transmission.max():.3f}]")
    print(f"   - Refined Transmission mean: {transmission.mean():.3f}")

    return transmission


# ============================================================================
# Scene Radiance Recovery (Image Formation Model - IFM)
# ============================================================================

def sceneRadianceRGB(img, transmission, AtomsphericLight):
    """
    Recover scene radiance using the Inverse Image Formation Model (IFM).

    Image Formation Model:
        I(x) = J(x) * t(x) + A * (1 - t(x))

    Where:
        I(x) = observed image (input)
        J(x) = scene radiance (what we want to recover)
        t(x) = transmission map
        A = atmospheric light

    Inverting this model:
        J(x) = (I(x) - A) / t(x) + A

    Args:
        img: Input degraded image
        transmission: Refined transmission map
        AtomsphericLight: Estimated atmospheric light

    Returns:
        sceneRadiance: Recovered clear image
    """
    print("[UDCP] Bước 6: Khôi phục ảnh bằng IFM (Image Formation Model)")
    print("   - Công thức IFM: J(x) = (I(x) - A) / t(x) + A")
    print(f"   - Atmospheric Light A: {AtomsphericLight}")

    AtomsphericLight = np.array(AtomsphericLight)
    img = np.float64(img)
    sceneRadiance = np.zeros(img.shape)

    # Clip transmission to avoid extreme values
    transmission = np.clip(transmission, 0.2, 0.9)
    print(f"   - Transmission được clip vào range: [0.2, 0.9]")

    # Apply IFM formula for each RGB channel
    for i in range(0, 3):
        sceneRadiance[:, :, i] = (img[:, :, i] - AtomsphericLight[i]) / transmission + AtomsphericLight[i]

    # Clip and convert to uint8
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    print(f"   - Scene Radiance shape: {sceneRadiance.shape}")
    print(f"   - Scene Radiance range: [{sceneRadiance.min()}, {sceneRadiance.max()}]")
    print("   - Đã khôi phục thành công ảnh rõ ràng!")

    return sceneRadiance


# ============================================================================
# Complete UDCP Pipeline
# ============================================================================

def process_udcp(img, blockSize=9):
    """
    Complete UDCP pipeline for underwater image enhancement.

    Args:
        img: Input underwater image (RGB, uint8)
        blockSize: Size of local patch for dark channel computation (default: 9)

    Returns:
        Dictionary containing:
            - 'output': Enhanced output image
            - 'dark_channel': Dark channel image
            - 'atmospheric_light': Estimated atmospheric light
            - 'transmission_raw': Initial transmission map
            - 'transmission_refined': Refined transmission map
    """
    print("\n" + "="*80)
    print("BẮT ĐẦU XỬ LÝ ẢNH BẰNG KỸ THUẬT UDCP")
    print("="*80)
    print(f"Input image shape: {img.shape}")
    print(f"Input image dtype: {img.dtype}")
    print(f"Input image range: [{img.min()}, {img.max()}]")
    print(f"Block size: {blockSize}x{blockSize}")
    print("="*80 + "\n")

    # Suppress overflow warnings
    np.seterr(over='ignore')

    # Step 1-2: Calculate Dark Channel
    GB_Darkchannel = getDarkChannel(img, blockSize)

    # Step 3: Estimate Atmospheric Light
    AtomsphericLight = getAtomsphericLight(GB_Darkchannel, img)

    # Step 4: Estimate Initial Transmission Map
    transmission_raw = getTransmission(img, AtomsphericLight, blockSize)

    # Step 5: Refine Transmission Map
    transmission_refined = Refinedtransmission(transmission_raw, img)

    # Step 6: Recover Scene Radiance
    sceneRadiance = sceneRadianceRGB(img, transmission_refined, AtomsphericLight)

    print("\n" + "="*80)
    print("HOÀN THÀNH XỬ LÝ UDCP")
    print("="*80)

    return {
        'output': sceneRadiance,
        'dark_channel': GB_Darkchannel,
        'atmospheric_light': AtomsphericLight,
        'transmission_raw': transmission_raw,
        'transmission_refined': transmission_refined
    }


# ============================================================================
# Main function for testing
# ============================================================================

if __name__ == '__main__':
    print("UDCP Module - Underwater Dark Channel Prior")
    print("This module should be imported and used by app.py")
    print("\nUsage example:")
    print("    from udcp import process_udcp")
    print("    img = cv2.imread('underwater_image.jpg')")
    print("    result = process_udcp(img, blockSize=9)")
    print("    output_img = result['output']")
