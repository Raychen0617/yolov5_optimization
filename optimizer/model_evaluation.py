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
    print("Model Speed : ", mean_syn)
    return mean_syn

def test_size(model, dummy_input):
    from torchsummary import summary
    summary(model, dummy_input)

def test_flops_and_params(model, dummy_input, device):

    # use fvcore for precise count
    # https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(model.to(device), dummy_input.to(device))
    print(flop_count_table(flops))
    

def evaluate_model(model, dummy_input, device,  testspeed, testflopsandparams):

    if testspeed: test_speed(model, dummy_input, device)
    if testflopsandparams: test_flops_and_params(model, dummy_input, device)
    #test_size(model, dummy_input)
