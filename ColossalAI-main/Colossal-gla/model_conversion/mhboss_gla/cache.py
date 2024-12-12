# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers


class Cache(transformers.cache_utils.Cache):
    """
    A cache used for storing hidden states produced by BoSs.

    It stores the states (hidden states and Key/Value/sid states) of each layer. The expected shape for each tensor of the main part is `[batch_size, key_dim, value_dim]`,
    and the expected shape for each tensor of the BoSs part is `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(
        self,
        seen_tokens: int = 0
    ) -> Cache:

        self.main_states: List[torch.Tensor] = []
        self.boss_states: List[torch.Tensor] = []
        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        if layer_idx < len(self):
            return (self.main_states[layer_idx], self.boss_states[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.main_states[layer_idx], self.boss_states[layer_idx])

    def __len__(self):
        return len(self.main_states)

    def update(
        self,
        state: Tuple[torch.Tensor],
        layer_idx: int,
        part: str,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the cache with the new `state` for the layer `layer_idx`.

        Parameters:
            state (`Tuple[torch.Tensor]`):
                The new state to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            part (`str`):
                Indicate states type (of BoSs part or the main part).
            offset (`int`):
                The offset of current fed tokens.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            The updated state.
        """

        if isinstance(state, torch.Tensor):
            state = (state,)
        if len(self.main_states) <= layer_idx and part == "main":
            self.main_states.append(state)
        elif len(self.boss_states) <= layer_idx and part == "boss":
            self.boss_states.append(state)
        elif part == "main":
            for i, s in enumerate(state):
                self.main_states[layer_idx][i].copy_(s)
        else:
            for i, s in enumerate(state):
                self.boss_states[layer_idx][i] = torch.cat([self.boss_states[layer_idx][i], s], dim=-1 if i==2 else -2)
        # update the number of seen tokens once we achieve the last layer
        if layer_idx == 31 and part == "main":
            self._seen_tokens += offset

        return state if part == "main" else self.boss_states[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.main_states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.main_states)):
            device = self.main_states[layer_idx].device
            self.main_states[layer_idx] = self.main_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.boss_states[layer_idx].device
            self.boss_states[layer_idx] = self.boss_states[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.main_states[layer_idx], self.boss_states[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        
        cls,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        seen_tokens: int = 0
    ) -> Cache:
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                # print(f"Convert with {layer_idx}")
                main_states, boss_states = past_key_values[layer_idx]
                cache.update(main_states, layer_idx, "main")
                cache.update(boss_states, layer_idx, "boss")
        return cache
