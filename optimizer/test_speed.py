import numpy as np
from statistics import mean
import torch

def test_speed(model, dummy_input, device):
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions=300
    timings=np.zeros((repetitions,1))
    
    #GPU-WARM-UP
    for _ in range(32):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("*************", mean_syn, "*****************")
    return mean_syn