"""Compatibility shim: use directional_resolution_inference instead."""
from src.utils.directional_resolution_inference import DirectionalResolutionInference
EllipticalResolutionFitting = DirectionalResolutionInference
