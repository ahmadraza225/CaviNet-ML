# Training configurations

| File | Use |
|---|---|
| `train.yaml` | The real training (`cavinet-ml train`): every value of section 11.4 of the scope document. Change a value only with the team's approval, and record why. |
| `ci.yaml` | The synthetic end-to-end test and `make rehearsal`: the same recipe and network, cut down to 1 epoch and a batch of 2 so it runs on a CPU in minutes. Never for the real dataset. |

Unknown settings and invalid values are refused. Each training run saves the configuration it
used (`work/runs/fold_<k>/config.yaml`), and the model bundle records it.
