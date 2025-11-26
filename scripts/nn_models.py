from torch import nn
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv, GINEConv, GATConv


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        assert num_layers >= 2, "Number of layers should be >= 2"

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = relu(x)
        return x

class GINE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, num_layers=3):
        super().__init__()
        assert num_layers >= 2, "Number of layers should be >= 2"

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers.append(GINEConv(nn=mlp1, edge_dim=edge_dim))

        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers.append(GINEConv(nn=mlp, edge_dim=edge_dim))

        mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.layers.append(GINEConv(nn=mlp_out, edge_dim=edge_dim))

    def forward(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < self.num_layers - 1:
                x = relu(x)
        return x

class EGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, heads=4, num_layers=3):
        super().__init__()
        assert num_layers >= 2, "Number of layers should be >= 2"

        self.num_layers = num_layers
        self.heads = heads

        self.layers = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()

        # input layer
        # node dimension expanded by heads
        self.layers.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=0.0,
                edge_dim=hidden_dim    # use edge embedding in attention
            )
        )

        # Edge MLP to transform edge_attr to hidden_dim
        self.edge_mlps.append(
            nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )

        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=0.0,
                    edge_dim=hidden_dim
                )
            )

            self.edge_mlps.append(
                nn.Sequential(
                    nn.Linear(edge_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )

        # ----- Output layer -----
        self.layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=output_dim,
                heads=1,
                concat=False,
                dropout=0.0,
                edge_dim=hidden_dim
            )
        )

        self.edge_mlps.append(
            nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )

    def forward(self, x, edge_index, edge_attr):
        for i, (layer, edge_mlp) in enumerate(zip(self.layers, self.edge_mlps)):
            e = edge_mlp(edge_attr)
            x = layer(x, edge_index, e)
            if i < self.num_layers - 1:
                x = relu(x)
        return x