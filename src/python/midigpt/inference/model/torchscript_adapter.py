"""ModelBase adapter for legacy TorchScript checkpoints.

TorchScript ScriptModules expose only their ``forward`` (and no Python
attributes for ``cfg``/``n_head``/etc.). Inference machinery (KV-cache setup,
context bound checks) expects the full ``ModelBase`` surface. This adapter
fills in the missing methods.

Construction prefers an explicit ``ts_config`` dict (n_head, n_layer, n_embd,
n_positions). When unavailable, the adapter probes the model's internal
``transformer.{wte, h, wpe}`` layout — this is intentionally GPT-2-shaped and
exists only to keep legacy checkpoints working.
"""

from __future__ import annotations

import torch

_CANDIDATE_N_HEAD = (8, 16, 12, 4)


class TorchScriptAdapter:
    """Wraps a jit.ScriptModule and exposes the ModelBase surface."""

    arch = "torchscript"
    encoder_config: dict | None = None

    def __init__(
        self,
        scripted,
        ts_config: dict | None = None,
    ):
        self._model = scripted
        if ts_config is None:
            ts_config = self._probe(scripted)
        self._cfg = ts_config

    # ------------------------------------------------------------------- #
    #  ModelBase interface
    # ------------------------------------------------------------------- #
    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def forward(self, input_ids, past_kv=None):
        return self._model(input_ids, past_kv) if past_kv is not None else self._model(input_ids)

    def make_empty_kv(self) -> tuple:
        c = self._cfg
        try:
            dev = next(self._model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        return tuple(
            (
                torch.zeros(1, c["n_head"], 0, c["head_dim"], device=dev),
                torch.zeros(1, c["n_head"], 0, c["head_dim"], device=dev),
            )
            for _ in range(c["n_layer"])
        )

    def kv_length(self, past_kv) -> int:
        if past_kv is None or len(past_kv) == 0:
            return 0
        return int(past_kv[0][0].shape[2])

    def kv_null_positions(self, past_kv, spans) -> None:
        if past_kv is None or not spans:
            return
        for k_c, v_c in past_kv:
            for s, e in spans:
                k_c[:, :, s:e, :] = -1e4
                v_c[:, :, s:e, :] = 0.0

    def max_context(self) -> int:
        return int(self._cfg["n_positions"])

    def parameters(self):
        return self._model.parameters()

    # ------------------------------------------------------------------- #
    #  Probe (legacy GPT-2-shaped TorchScript)
    # ------------------------------------------------------------------- #
    @staticmethod
    def _probe(scripted) -> dict:
        trf = scripted.transformer
        n_embd = int(trf.wte.weight.shape[1])
        n_layer = sum(1 for _ in trf.h.children())
        n_positions = int(trf.wpe.weight.shape[0]) if hasattr(trf, "wpe") else 2048

        for n_head in _CANDIDATE_N_HEAD:
            if n_embd % n_head != 0:
                continue
            head_dim = n_embd // n_head
            kv = tuple(
                (torch.zeros(1, n_head, 0, head_dim), torch.zeros(1, n_head, 0, head_dim))
                for _ in range(n_layer)
            )
            try:
                with torch.no_grad():
                    scripted(torch.tensor([[0]], dtype=torch.long), kv)
                return {
                    "n_head": n_head,
                    "n_layer": n_layer,
                    "n_embd": n_embd,
                    "head_dim": head_dim,
                    "n_positions": n_positions,
                }
            except Exception:
                continue
        raise RuntimeError(
            "TorchScriptAdapter: could not infer model layout from scripted module. "
            "Provide ts_config={'n_head', 'n_layer', 'n_embd', 'head_dim', 'n_positions'} "
            "or migrate the checkpoint to the packed bundle format."
        )
