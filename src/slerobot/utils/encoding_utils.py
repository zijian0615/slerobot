# Copyright 2024 The HuggingFace Inc. team. All rights reserved.


def encode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    """Encode a signed integer into sign-magnitude representation for Feetech registers."""
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(int(value))
    if magnitude > max_magnitude:
        raise ValueError(
            f"Magnitude {magnitude} exceeds {max_magnitude} (max for sign_bit_index={sign_bit_index})"
        )
    encoded = magnitude
    if value < 0:
        encoded |= 1 << sign_bit_index
    return encoded


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int) -> int:
    """Decode sign-magnitude representation into a signed integer."""
    sign_bit = 1 << sign_bit_index
    magnitude = int(encoded_value) & (sign_bit - 1)
    if int(encoded_value) & sign_bit:
        return -magnitude
    return magnitude
