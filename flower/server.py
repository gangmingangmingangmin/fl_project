import flwr as fl
from typing import Callable, Dict

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    def fit_config(rnd:int) -> Dict[str,str]:
        config = {"epoch" : 1}
        return config
    return fit_config


strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,  # Sample 10% of available clients for the next round
    min_fit_clients=2,  # Minimum number of clients to be sampled for the next round
    min_available_clients=9,  # Minimum number of clients that need to be connected to the server before a training round can start
    on_fit_config_fn=get_on_fit_config_fn()
)



fl.server.start_server(config={"num_rounds": 3},strategy=strategy)
