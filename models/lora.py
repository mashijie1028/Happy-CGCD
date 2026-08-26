import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# defaults for LoRA
_DEFAULT_LORA_RANK = 64
_DEFAULT_LORA_ALPHA = None
_DEFAULT_LORA_DROPOUT = 0.0


class Linear(nn.Module):
    """
    A drop-in replacement for nn.Linear with extra support for LoRA (Low-Rank Adaptation).

    LoRA is **lazy**, call `init_lora()` to enable, otherwise it is same as `nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        # default to require_grad=True, as in nn.Linear
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        # LoRA layers; and its weights (1.0 if not specified)
        self.lora_layers = nn.ParameterDict()
        self.lora_weights: dict[str, float] = {}

    def init_lora(
        self,
        layer_id: int | str,
        layer_weight: int | float = None,
        r: int = _DEFAULT_LORA_RANK,
        alpha: float = _DEFAULT_LORA_ALPHA,
        dropout: float = _DEFAULT_LORA_DROPOUT,
    ):
        """
        Initialize one layer of LoRA parameters.
        This method can be called to reinitialize LoRA parameters if needed.
        """
        layer_id = f"{layer_id}"

        if layer_id in self.lora_layers.keys():
            # if exists, skip re-init, print to stderr
            print(
                f"WARN: LoRA layer {layer_id} exists {self.lora_layers.keys()}, skip re-init.",
                file=sys.stderr,
            )
            return
        assert r > 0, f"LoRA rank {r} must be > 0"

        out_features, in_features = self.weight.shape
        lora_term = LoRA_BA_Term(
            in_features=in_features,
            out_features=out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        # disable w, b, other lora; enable the newly added one
        self.requires_grad_(False)
        lora_term.requires_grad_(True)

        self.lora_layers.add_module(layer_id, lora_term)
        if layer_weight is not None:
            assert isinstance(layer_weight, (int, float))
            self.lora_weights[layer_id] = layer_weight

    def requires_grad_(self, mode: bool = True):
        self.weight.requires_grad_(mode)
        self.bias.requires_grad_(mode)
        for m in self.lora_layers.children():
            m: LoRA_BA_Term
            m.requires_grad_(mode)

    def merge_lora(self, layer_id: int | str):
        layer_id = f"{layer_id}"
        assert layer_id in self.lora_layers

        # merges lora to nn.linear
        lora_layer: LoRA_BA_Term = self.lora_layers[layer_id]
        if lora_layer.r == 0:
            return

        # merge the LoRA parameters into the main weight
        with torch.no_grad():
            self.weight += lora_layer.a @ lora_layer.b.T * lora_layer.scaling

        # remove the LoRA layer
        del self.lora_layers[layer_id]

    def forward(
        self,
        x: torch.Tensor,
        override_lora_weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        r"""
        Forward pass for the linear layer with optional LoRA weights.

        Output a weighted sum of the main linear layer and all LoRA layers.
        $W(x) + \sum{} \text{lora\_weights}_i \cdot \text{lora\_layers}_i(x)$
        LoRA layers are ordered FIFO.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        lora_weights : torch.Tensor, optional
            Weights for LoRA, numel==n.lora_layers, by default 1:..:1.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # computes a weighted sum of all LoRA layers
        # default linear:lora1:...:loraN = 1:1:...:1

        # same as nn.Linear, if no LoRA available
        result = F.linear(x, self.weight, bias=self.bias)
        if len(self.lora_layers) == 0:
            return result

        if override_lora_weights is not None:
            lora_weights = override_lora_weights
        else:
            lora_weights = self.lora_weights

        # cal weighted sum of all LoRA layers, default weight 1.0
        for layer_id, lora_layer in self.lora_layers.items():
            lora_layer: LoRA_BA_Term

            # if no weight specified, use 1.0
            result += lora_weights.get(layer_id, 1.0) * lora_layer(x)

        return result


class LoRA_BA_Term(nn.Module):
    """
    A linear layer that implements the LoRA (Low-Rank Adaptation) technique. \n
    Only includes the addition term, i.e. BA, without W.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = _DEFAULT_LORA_RANK,
        alpha: float = _DEFAULT_LORA_ALPHA,
        dropout: float = _DEFAULT_LORA_DROPOUT,
    ):
        super().__init__()
        self.r = r
        self.scaling = 1.0 if alpha is None else alpha / r

        # $\Delta W=BA$, dropout
        self.a = nn.Parameter(torch.empty((r, in_features)), requires_grad=True)
        self.b = nn.Parameter(torch.empty((out_features, r)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

        # init params
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.a, a=5**0.5)
            nn.init.zeros_(self.b)

    def requires_grad_(self, mode: bool = True):
        self.a.requires_grad_(mode)
        self.b.requires_grad_(mode)
        self.dropout.train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.a.T @ self.b.T) * self.scaling
