#!/usr/bin/env python
"""
Emotion Hooks - Integration Points for PAD Prediction

Provides hooks for integrating PAD emotion predictions into:
- Training loops (emotion-aware learning rate/temperature)
- Inference pipelines (emotion-conditioned generation)
- Multimodal fusion (PAD-gated modality weighting)
- Monitoring dashboards (real-time emotion tracking)

Hook types:
- pre_forward: Before model forward pass
- post_forward: After model forward pass
- pre_optimizer_step: Before optimizer update
- post_batch: After processing batch
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict

from .pad_regressor import PADRegressor, PADPrediction


@dataclass
class EmotionState:
    """Current emotion state with history."""
    pad: PADPrediction
    timestamp: float
    modality_sources: Dict[str, bool]  # Which modalities were used


class EmotionHookRegistry:
    """
    Registry for emotion-based hooks.

    Allows registration of callbacks for different events:
    - PAD prediction computed
    - Emotion-aware parameter adjustment
    - Emotion state change detection
    """

    def __init__(self):
        self.hooks = defaultdict(list)

    def register(
        self,
        event: str,
        callback: Callable,
        priority: int = 0,
    ):
        """
        Register a hook callback.

        Args:
            event: Event name ("pre_forward", "post_forward", etc.)
            callback: Callback function
            priority: Priority (higher = called first)
        """
        self.hooks[event].append((priority, callback))
        self.hooks[event].sort(key=lambda x: -x[0])  # Sort by priority descending

    def trigger(
        self,
        event: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Trigger all hooks for an event.

        Args:
            event: Event name
            *args, **kwargs: Arguments to pass to callbacks

        Returns:
            Results from all callbacks
        """
        results = []

        for priority, callback in self.hooks.get(event, []):
            result = callback(*args, **kwargs)
            results.append(result)

        return results

    def clear(self, event: Optional[str] = None):
        """Clear hooks for an event or all events."""
        if event is None:
            self.hooks.clear()
        else:
            self.hooks[event].clear()


class PADEmotionHook(nn.Module):
    """
    PAD Emotion Hook Module.

    Integrates PAD prediction into model forward pass with callbacks.
    """

    def __init__(
        self,
        pad_regressor: Optional[PADRegressor] = None,
        enable_emotion_aware_lr: bool = True,
        enable_emotion_aware_temp: bool = True,
        enable_pad_caching: bool = True,
    ):
        """
        Initialize PAD emotion hook.

        Args:
            pad_regressor: PAD regressor module (if None, created internally)
            enable_emotion_aware_lr: Enable emotion-aware learning rate
            enable_emotion_aware_temp: Enable emotion-aware temperature
            enable_pad_caching: Cache PAD predictions to avoid recomputation
        """
        super().__init__()

        # PAD regressor
        if pad_regressor is None:
            pad_regressor = PADRegressor()

        self.pad_regressor = pad_regressor

        # Configuration
        self.enable_emotion_aware_lr = enable_emotion_aware_lr
        self.enable_emotion_aware_temp = enable_emotion_aware_temp
        self.enable_pad_caching = enable_pad_caching

        # Hook registry
        self.registry = EmotionHookRegistry()

        # State tracking
        self.current_state: Optional[EmotionState] = None
        self.state_history = []

        # PAD cache (to avoid recomputation)
        self.pad_cache = {}

        # Emotion-aware parameters
        self.base_lr = 1.0
        self.base_temp = 1.0

        # Register default hooks
        self._register_default_hooks()

    def _register_default_hooks(self):
        """Register default emotion-aware hooks."""

        # Hook 1: Emotion-aware learning rate
        if self.enable_emotion_aware_lr:
            def adjust_lr(state: EmotionState, optimizer: torch.optim.Optimizer):
                """
                Adjust learning rate based on arousal.

                High arousal → higher LR (more exploration)
                Low arousal → lower LR (more exploitation)
                """
                arousal = state.pad.arousal.mean().item()

                # Scale LR by arousal
                lr_scale = 0.5 + arousal  # Range [0.5, 1.5]

                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.base_lr * lr_scale

                return lr_scale

            self.registry.register("pre_optimizer_step", adjust_lr, priority=10)

        # Hook 2: Emotion-aware temperature
        if self.enable_emotion_aware_temp:
            def adjust_temperature(state: EmotionState) -> float:
                """
                Adjust sampling temperature based on pleasure and dominance.

                High pleasure → lower temp (more confident)
                Low dominance → higher temp (more exploration)
                """
                pleasure = state.pad.pleasure.mean().item()
                dominance = state.pad.dominance.mean().item()

                # Compute temperature scale
                temp_scale = 1.0 - (pleasure * 0.2) + (1.0 - dominance) * 0.3

                return self.base_temp * temp_scale

            self.registry.register("post_forward", adjust_temperature, priority=5)

    def predict_pad(
        self,
        audio: Optional[torch.Tensor] = None,
        frames: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> PADPrediction:
        """
        Predict PAD from inputs.

        Args:
            audio: Audio tensor
            frames: Video frames
            use_cache: Use cached prediction if available

        Returns:
            PAD prediction
        """
        # Create cache key
        cache_key = None
        if use_cache and self.enable_pad_caching:
            cache_key = (
                id(audio) if audio is not None else None,
                id(frames) if frames is not None else None,
            )

            if cache_key in self.pad_cache:
                return self.pad_cache[cache_key]

        # Predict PAD
        with torch.no_grad():
            pad_pred = self.pad_regressor(
                audio=audio,
                frames=frames,
                return_modality_contributions=True,
            )

        # Cache result
        if cache_key is not None:
            self.pad_cache[cache_key] = pad_pred

        return pad_pred

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        frames: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EmotionState:
        """
        Run PAD prediction and trigger hooks.

        Args:
            audio: Audio input
            frames: Video frames
            **kwargs: Additional arguments for hooks

        Returns:
            EmotionState with PAD prediction
        """
        import time

        # Predict PAD
        pad_pred = self.predict_pad(audio, frames)

        # Create emotion state
        state = EmotionState(
            pad=pad_pred,
            timestamp=time.time(),
            modality_sources={
                "audio": audio is not None,
                "face": frames is not None,
            },
        )

        # Update current state
        self.current_state = state
        self.state_history.append(state)

        # Trigger post-forward hooks
        self.registry.trigger("post_forward", state, **kwargs)

        return state

    def pre_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ):
        """
        Trigger pre-optimizer-step hooks.

        Args:
            optimizer: Optimizer instance
            **kwargs: Additional arguments
        """
        if self.current_state is not None:
            self.registry.trigger("pre_optimizer_step", self.current_state, optimizer, **kwargs)

    def post_batch(self, **kwargs):
        """Trigger post-batch hooks."""
        if self.current_state is not None:
            self.registry.trigger("post_batch", self.current_state, **kwargs)

    def get_current_pad(self) -> Optional[PADPrediction]:
        """Get current PAD prediction."""
        if self.current_state is not None:
            return self.current_state.pad
        return None

    def get_pad_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of recent PAD predictions.

        Returns:
            Dict with mean P/A/D values
        """
        if not self.state_history:
            return {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0}

        # Compute mean over recent history (last 10)
        recent = self.state_history[-10:]

        pleasure_values = [s.pad.pleasure.mean().item() for s in recent]
        arousal_values = [s.pad.arousal.mean().item() for s in recent]
        dominance_values = [s.pad.dominance.mean().item() for s in recent]

        return {
            "pleasure": sum(pleasure_values) / len(pleasure_values),
            "arousal": sum(arousal_values) / len(arousal_values),
            "dominance": sum(dominance_values) / len(dominance_values),
        }

    def clear_cache(self):
        """Clear PAD cache."""
        self.pad_cache.clear()

    def clear_history(self):
        """Clear state history."""
        self.state_history.clear()


def create_emotion_aware_trainer_hook(
    pad_regressor: Optional[PADRegressor] = None,
) -> PADEmotionHook:
    """
    Create PAD emotion hook for training.

    Args:
        pad_regressor: Optional PAD regressor

    Returns:
        Configured PADEmotionHook
    """
    return PADEmotionHook(
        pad_regressor=pad_regressor,
        enable_emotion_aware_lr=True,
        enable_emotion_aware_temp=True,
        enable_pad_caching=True,
    )


def create_emotion_aware_inference_hook(
    pad_regressor: Optional[PADRegressor] = None,
) -> PADEmotionHook:
    """
    Create PAD emotion hook for inference.

    Args:
        pad_regressor: Optional PAD regressor

    Returns:
        Configured PADEmotionHook (without LR adjustment)
    """
    return PADEmotionHook(
        pad_regressor=pad_regressor,
        enable_emotion_aware_lr=False,  # No LR in inference
        enable_emotion_aware_temp=True,
        enable_pad_caching=True,
    )
