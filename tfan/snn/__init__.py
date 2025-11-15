"""
GPU-Emulated Spiking Neural Network (SNN) for TF-A-N.

Core components for building and training SNNs with TF-A-N integration.

Modules:
- neuron: LIF, PLIF, Izhikevich neurons with surrogate gradients
- layers: SpikingLinear, SpikingConv2d, SpikingResidualBlock
- encode: Rate, latency, delta encoders
- readout: Spike-count, membrane, CTC readouts
"""

from .neuron import (
    LIF,
    PLIF,
    IzhikevichNeuron,
    NeuronState,
    get_surrogate_fn,
)

from .layers import (
    SpikingLinear,
    SpikingConv2d,
    SpikingResidualBlock,
    SpikingSelfAttention,
)

from .encode import (
    RateEncoder,
    LatencyEncoder,
    DeltaEncoder,
    create_encoder,
)

from .readout import (
    SpikeCountReadout,
    MembraneReadout,
    CTCReadout,
)

__version__ = "0.1.0"

__all__ = [
    # Neurons
    "LIF",
    "PLIF",
    "IzhikevichNeuron",
    "NeuronState",
    "get_surrogate_fn",
    # Layers
    "SpikingLinear",
    "SpikingConv2d",
    "SpikingResidualBlock",
    "SpikingSelfAttention",
    # Encoders
    "RateEncoder",
    "LatencyEncoder",
    "DeltaEncoder",
    "create_encoder",
    # Readouts
    "SpikeCountReadout",
    "MembraneReadout",
    "CTCReadout",
]
