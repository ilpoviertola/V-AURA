import torch


class BridgeBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.bridge(x)
            return x
        except TypeError as e:
            raise TypeError(
                "The class cant be called on its own. Please, use a class that inherits it",
                e,
            )


class ConvBridgeBase(BridgeBase):
    def __init__(self, block, **kwargs) -> None:
        super().__init__()
        self.bridge = torch.nn.Sequential(
            block(**kwargs),
            torch.nn.GELU(),
        )


class ConvBridgeVisual(ConvBridgeBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(block=torch.nn.Conv3d, **kwargs)


class ConvBridge2D(ConvBridgeBase):

    def __init__(self, **kwargs) -> None:
        super().__init__(block=torch.nn.Conv2d, **kwargs)


class MLPBridge(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        if activation == "relu":
            activation = torch.nn.ReLU()
        elif activation == "gelu":
            activation = torch.nn.GELU()
        else:
            raise ValueError(f"Activation {activation} not supported")

        self.bridge = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            activation,
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bridge(x)
