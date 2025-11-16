#!/usr/bin/env python
"""PAD Trainer Hooks - integrate PAD feedback into training."""

import torch


def on_batch_metrics(pad_vec, trainer):
    """
    Hook called after PAD estimation.

    Modulates trainer temperature and attention keep_ratio based on emotion.

    Args:
        pad_vec: PAD vector [batch, 3]
        trainer: Trainer instance with pad_gate, scheduler, attn
    """
    T, k = trainer.pad_gate.schedule(pad_vec)

    # Set temperature (averaged over batch)
    trainer.scheduler.set_temperature(T.mean().item())

    # Set keep ratio for sparse attention
    trainer.attn.set_keep_ratio(k.mean().item())


class PADTrainerHook:
    """Trainer hook for PAD-based modulation."""

    def __init__(self, pad_gate, pad_regressor):
        self.pad_gate = pad_gate
        self.pad_regressor = pad_regressor

    def on_batch(self, audio, face, trainer):
        """Process batch and update trainer params."""
        # Extract features and predict PAD
        audio_feat = trainer.audio_fe(audio)
        face_feat = trainer.face_fe(face)
        pad_vec = self.pad_regressor(audio_feat, face_feat)

        # Apply modulation
        on_batch_metrics(pad_vec, trainer)

        return pad_vec
