
import torch
import torch.nn
import torch.functional
import torch.nn.functional
import torch.quantization
import torch.nn.quantized


class Model_qat(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fake_quant_0 = torch.quantization.QuantStub()
        self.model_0_conv = torch.nn.Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
        self.model_0_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_0_act = torch.nn.SiLU(inplace=True)
        self.model_1_conv = torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.model_1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_1_act = torch.nn.SiLU(inplace=True)
        self.model_2_cv1_conv = torch.nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_2_cv1_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_2_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_2_m_0_cv1_conv = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_2_m_0_cv1_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_2_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_2_m_0_cv2_conv = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_2_m_0_cv2_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_2_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_2_cv2_conv = torch.nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_2_cv2_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_2_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_2_cv3_conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_2_cv3_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_2_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_3_conv = torch.nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.model_3_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_3_act = torch.nn.SiLU(inplace=True)
        self.model_4_cv1_conv = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_4_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_4_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_4_m_0_cv1_conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_m_0_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_4_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_4_m_0_cv2_conv = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_4_m_0_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_4_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_4_m_1_cv1_conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_m_1_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_4_m_1_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_4_m_1_cv2_conv = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_4_m_1_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_4_m_1_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_4_cv2_conv = torch.nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_4_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_4_cv3_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_cv3_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_4_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_5_conv = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_5_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_5_act = torch.nn.SiLU(inplace=True)
        self.model_6_cv1_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_6_m_0_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_m_0_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_6_m_0_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_6_m_0_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_6_m_1_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_m_1_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_m_1_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_6_m_1_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_6_m_1_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_m_1_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_6_m_2_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_m_2_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_m_2_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_6_m_2_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_6_m_2_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_m_2_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_6_cv2_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_6_cv3_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_cv3_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_6_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_7_conv = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_7_bn = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_7_act = torch.nn.SiLU(inplace=True)
        self.model_8_cv1_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_8_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_8_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_8_m_0_cv1_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_8_m_0_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_8_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_8_m_0_cv2_conv = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_8_m_0_cv2_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_8_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_8_cv2_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_8_cv2_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_8_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_8_cv3_conv = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_8_cv3_bn = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_8_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_9_cv1_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_9_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_9_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_9_m = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.model_9_m_1 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.model_9_m_2 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.model_9_cv2_conv = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_9_cv2_bn = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_9_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_10_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_10_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_10_act = torch.nn.SiLU(inplace=True)
        self.model_11 = torch.nn.Upsample(scale_factor=2.0)
        self.model_13_cv1_conv = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_13_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_13_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_13_m_0_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_13_m_0_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_13_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_13_m_0_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_13_m_0_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_13_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_13_cv2_conv = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_13_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_13_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_13_cv3_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_13_cv3_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_13_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_14_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_14_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_14_act = torch.nn.SiLU(inplace=True)
        self.model_15 = torch.nn.Upsample(scale_factor=2.0)
        self.model_17_cv1_conv = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_17_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_17_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_17_m_0_cv1_conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_17_m_0_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_17_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_17_m_0_cv2_conv = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_17_m_0_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_17_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_17_cv2_conv = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_17_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_17_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_17_cv3_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_17_cv3_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_17_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_18_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_18_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_18_act = torch.nn.SiLU(inplace=True)
        self.model_20_cv1_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_20_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_20_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_20_m_0_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_20_m_0_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_20_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_20_m_0_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_20_m_0_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_20_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_20_cv2_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_20_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_20_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_20_cv3_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_20_cv3_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_20_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_21_conv = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_21_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_21_act = torch.nn.SiLU(inplace=True)
        self.model_23_cv1_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_23_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_23_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_23_m_0_cv1_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_23_m_0_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_23_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.model_23_m_0_cv2_conv = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_23_m_0_cv2_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_23_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_23_cv2_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_23_cv2_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_23_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_23_cv3_conv = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_23_cv3_bn = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.model_23_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_24_m_0 = torch.nn.Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        self.model_24_m_1 = torch.nn.Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        self.model_24_m_2 = torch.nn.Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        self.fake_dequant_0 = torch.quantization.DeQuantStub()
        self.fake_dequant_1 = torch.quantization.DeQuantStub()
        self.fake_dequant_2 = torch.quantization.DeQuantStub()
        self.float_functional_simple_0 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_1 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_2 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_3 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_4 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_5 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_6 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_7 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_8 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_9 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_10 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_11 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_12 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_13 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_14 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_15 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_16 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_17 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_18 = torch.nn.quantized.FloatFunctional()
        self.float_functional_simple_19 = torch.nn.quantized.FloatFunctional()

    def forward(self, input_0_f):
        fake_quant_0 = self.fake_quant_0(input_0_f)
        input_0_f = None
        model_0_conv = self.model_0_conv(fake_quant_0)
        fake_quant_0 = None
        model_0_bn = self.model_0_bn(model_0_conv)
        model_0_conv = None
        model_0_act = self.model_0_act(model_0_bn)
        model_0_bn = None
        model_1_conv = self.model_1_conv(model_0_act)
        model_0_act = None
        model_1_bn = self.model_1_bn(model_1_conv)
        model_1_conv = None
        model_1_act = self.model_1_act(model_1_bn)
        model_1_bn = None
        model_2_cv1_conv = self.model_2_cv1_conv(model_1_act)
        model_2_cv1_bn = self.model_2_cv1_bn(model_2_cv1_conv)
        model_2_cv1_conv = None
        model_2_cv1_act = self.model_2_cv1_act(model_2_cv1_bn)
        model_2_cv1_bn = None
        model_2_m_0_cv1_conv = self.model_2_m_0_cv1_conv(model_2_cv1_act)
        model_2_m_0_cv1_bn = self.model_2_m_0_cv1_bn(model_2_m_0_cv1_conv)
        model_2_m_0_cv1_conv = None
        model_2_m_0_cv1_act = self.model_2_m_0_cv1_act(model_2_m_0_cv1_bn)
        model_2_m_0_cv1_bn = None
        model_2_m_0_cv2_conv = self.model_2_m_0_cv2_conv(model_2_m_0_cv1_act)
        model_2_m_0_cv1_act = None
        model_2_m_0_cv2_bn = self.model_2_m_0_cv2_bn(model_2_m_0_cv2_conv)
        model_2_m_0_cv2_conv = None
        model_2_m_0_cv2_act = self.model_2_m_0_cv2_act(model_2_m_0_cv2_bn)
        model_2_m_0_cv2_bn = None
        add_0_f = self.float_functional_simple_0.add(model_2_cv1_act, model_2_m_0_cv2_act)
        model_2_cv1_act = None
        model_2_m_0_cv2_act = None
        model_2_cv2_conv = self.model_2_cv2_conv(model_1_act)
        model_1_act = None
        model_2_cv2_bn = self.model_2_cv2_bn(model_2_cv2_conv)
        model_2_cv2_conv = None
        model_2_cv2_act = self.model_2_cv2_act(model_2_cv2_bn)
        model_2_cv2_bn = None
        cat_0_f = self.float_functional_simple_1.cat([add_0_f, model_2_cv2_act], 1)
        add_0_f = None
        model_2_cv2_act = None
        model_2_cv3_conv = self.model_2_cv3_conv(cat_0_f)
        cat_0_f = None
        model_2_cv3_bn = self.model_2_cv3_bn(model_2_cv3_conv)
        model_2_cv3_conv = None
        model_2_cv3_act = self.model_2_cv3_act(model_2_cv3_bn)
        model_2_cv3_bn = None
        model_3_conv = self.model_3_conv(model_2_cv3_act)
        model_2_cv3_act = None
        model_3_bn = self.model_3_bn(model_3_conv)
        model_3_conv = None
        model_3_act = self.model_3_act(model_3_bn)
        model_3_bn = None
        model_4_cv1_conv = self.model_4_cv1_conv(model_3_act)
        model_4_cv1_bn = self.model_4_cv1_bn(model_4_cv1_conv)
        model_4_cv1_conv = None
        model_4_cv1_act = self.model_4_cv1_act(model_4_cv1_bn)
        model_4_cv1_bn = None
        model_4_m_0_cv1_conv = self.model_4_m_0_cv1_conv(model_4_cv1_act)
        model_4_m_0_cv1_bn = self.model_4_m_0_cv1_bn(model_4_m_0_cv1_conv)
        model_4_m_0_cv1_conv = None
        model_4_m_0_cv1_act = self.model_4_m_0_cv1_act(model_4_m_0_cv1_bn)
        model_4_m_0_cv1_bn = None
        model_4_m_0_cv2_conv = self.model_4_m_0_cv2_conv(model_4_m_0_cv1_act)
        model_4_m_0_cv1_act = None
        model_4_m_0_cv2_bn = self.model_4_m_0_cv2_bn(model_4_m_0_cv2_conv)
        model_4_m_0_cv2_conv = None
        model_4_m_0_cv2_act = self.model_4_m_0_cv2_act(model_4_m_0_cv2_bn)
        model_4_m_0_cv2_bn = None
        add_1_f = self.float_functional_simple_2.add(model_4_cv1_act, model_4_m_0_cv2_act)
        model_4_cv1_act = None
        model_4_m_0_cv2_act = None
        model_4_m_1_cv1_conv = self.model_4_m_1_cv1_conv(add_1_f)
        model_4_m_1_cv1_bn = self.model_4_m_1_cv1_bn(model_4_m_1_cv1_conv)
        model_4_m_1_cv1_conv = None
        model_4_m_1_cv1_act = self.model_4_m_1_cv1_act(model_4_m_1_cv1_bn)
        model_4_m_1_cv1_bn = None
        model_4_m_1_cv2_conv = self.model_4_m_1_cv2_conv(model_4_m_1_cv1_act)
        model_4_m_1_cv1_act = None
        model_4_m_1_cv2_bn = self.model_4_m_1_cv2_bn(model_4_m_1_cv2_conv)
        model_4_m_1_cv2_conv = None
        model_4_m_1_cv2_act = self.model_4_m_1_cv2_act(model_4_m_1_cv2_bn)
        model_4_m_1_cv2_bn = None
        add_2_f = self.float_functional_simple_3.add(add_1_f, model_4_m_1_cv2_act)
        add_1_f = None
        model_4_m_1_cv2_act = None
        model_4_cv2_conv = self.model_4_cv2_conv(model_3_act)
        model_3_act = None
        model_4_cv2_bn = self.model_4_cv2_bn(model_4_cv2_conv)
        model_4_cv2_conv = None
        model_4_cv2_act = self.model_4_cv2_act(model_4_cv2_bn)
        model_4_cv2_bn = None
        cat_1_f = self.float_functional_simple_4.cat([add_2_f, model_4_cv2_act], 1)
        add_2_f = None
        model_4_cv2_act = None
        model_4_cv3_conv = self.model_4_cv3_conv(cat_1_f)
        cat_1_f = None
        model_4_cv3_bn = self.model_4_cv3_bn(model_4_cv3_conv)
        model_4_cv3_conv = None
        model_4_cv3_act = self.model_4_cv3_act(model_4_cv3_bn)
        model_4_cv3_bn = None
        model_5_conv = self.model_5_conv(model_4_cv3_act)
        model_5_bn = self.model_5_bn(model_5_conv)
        model_5_conv = None
        model_5_act = self.model_5_act(model_5_bn)
        model_5_bn = None
        model_6_cv1_conv = self.model_6_cv1_conv(model_5_act)
        model_6_cv1_bn = self.model_6_cv1_bn(model_6_cv1_conv)
        model_6_cv1_conv = None
        model_6_cv1_act = self.model_6_cv1_act(model_6_cv1_bn)
        model_6_cv1_bn = None
        model_6_m_0_cv1_conv = self.model_6_m_0_cv1_conv(model_6_cv1_act)
        model_6_m_0_cv1_bn = self.model_6_m_0_cv1_bn(model_6_m_0_cv1_conv)
        model_6_m_0_cv1_conv = None
        model_6_m_0_cv1_act = self.model_6_m_0_cv1_act(model_6_m_0_cv1_bn)
        model_6_m_0_cv1_bn = None
        model_6_m_0_cv2_conv = self.model_6_m_0_cv2_conv(model_6_m_0_cv1_act)
        model_6_m_0_cv1_act = None
        model_6_m_0_cv2_bn = self.model_6_m_0_cv2_bn(model_6_m_0_cv2_conv)
        model_6_m_0_cv2_conv = None
        model_6_m_0_cv2_act = self.model_6_m_0_cv2_act(model_6_m_0_cv2_bn)
        model_6_m_0_cv2_bn = None
        add_3_f = self.float_functional_simple_5.add(model_6_cv1_act, model_6_m_0_cv2_act)
        model_6_cv1_act = None
        model_6_m_0_cv2_act = None
        model_6_m_1_cv1_conv = self.model_6_m_1_cv1_conv(add_3_f)
        model_6_m_1_cv1_bn = self.model_6_m_1_cv1_bn(model_6_m_1_cv1_conv)
        model_6_m_1_cv1_conv = None
        model_6_m_1_cv1_act = self.model_6_m_1_cv1_act(model_6_m_1_cv1_bn)
        model_6_m_1_cv1_bn = None
        model_6_m_1_cv2_conv = self.model_6_m_1_cv2_conv(model_6_m_1_cv1_act)
        model_6_m_1_cv1_act = None
        model_6_m_1_cv2_bn = self.model_6_m_1_cv2_bn(model_6_m_1_cv2_conv)
        model_6_m_1_cv2_conv = None
        model_6_m_1_cv2_act = self.model_6_m_1_cv2_act(model_6_m_1_cv2_bn)
        model_6_m_1_cv2_bn = None
        add_4_f = self.float_functional_simple_6.add(add_3_f, model_6_m_1_cv2_act)
        add_3_f = None
        model_6_m_1_cv2_act = None
        model_6_m_2_cv1_conv = self.model_6_m_2_cv1_conv(add_4_f)
        model_6_m_2_cv1_bn = self.model_6_m_2_cv1_bn(model_6_m_2_cv1_conv)
        model_6_m_2_cv1_conv = None
        model_6_m_2_cv1_act = self.model_6_m_2_cv1_act(model_6_m_2_cv1_bn)
        model_6_m_2_cv1_bn = None
        model_6_m_2_cv2_conv = self.model_6_m_2_cv2_conv(model_6_m_2_cv1_act)
        model_6_m_2_cv1_act = None
        model_6_m_2_cv2_bn = self.model_6_m_2_cv2_bn(model_6_m_2_cv2_conv)
        model_6_m_2_cv2_conv = None
        model_6_m_2_cv2_act = self.model_6_m_2_cv2_act(model_6_m_2_cv2_bn)
        model_6_m_2_cv2_bn = None
        add_5_f = self.float_functional_simple_7.add(add_4_f, model_6_m_2_cv2_act)
        add_4_f = None
        model_6_m_2_cv2_act = None
        model_6_cv2_conv = self.model_6_cv2_conv(model_5_act)
        model_5_act = None
        model_6_cv2_bn = self.model_6_cv2_bn(model_6_cv2_conv)
        model_6_cv2_conv = None
        model_6_cv2_act = self.model_6_cv2_act(model_6_cv2_bn)
        model_6_cv2_bn = None
        cat_2_f = self.float_functional_simple_8.cat([add_5_f, model_6_cv2_act], 1)
        add_5_f = None
        model_6_cv2_act = None
        model_6_cv3_conv = self.model_6_cv3_conv(cat_2_f)
        cat_2_f = None
        model_6_cv3_bn = self.model_6_cv3_bn(model_6_cv3_conv)
        model_6_cv3_conv = None
        model_6_cv3_act = self.model_6_cv3_act(model_6_cv3_bn)
        model_6_cv3_bn = None
        model_7_conv = self.model_7_conv(model_6_cv3_act)
        model_7_bn = self.model_7_bn(model_7_conv)
        model_7_conv = None
        model_7_act = self.model_7_act(model_7_bn)
        model_7_bn = None
        model_8_cv1_conv = self.model_8_cv1_conv(model_7_act)
        model_8_cv1_bn = self.model_8_cv1_bn(model_8_cv1_conv)
        model_8_cv1_conv = None
        model_8_cv1_act = self.model_8_cv1_act(model_8_cv1_bn)
        model_8_cv1_bn = None
        model_8_m_0_cv1_conv = self.model_8_m_0_cv1_conv(model_8_cv1_act)
        model_8_m_0_cv1_bn = self.model_8_m_0_cv1_bn(model_8_m_0_cv1_conv)
        model_8_m_0_cv1_conv = None
        model_8_m_0_cv1_act = self.model_8_m_0_cv1_act(model_8_m_0_cv1_bn)
        model_8_m_0_cv1_bn = None
        model_8_m_0_cv2_conv = self.model_8_m_0_cv2_conv(model_8_m_0_cv1_act)
        model_8_m_0_cv1_act = None
        model_8_m_0_cv2_bn = self.model_8_m_0_cv2_bn(model_8_m_0_cv2_conv)
        model_8_m_0_cv2_conv = None
        model_8_m_0_cv2_act = self.model_8_m_0_cv2_act(model_8_m_0_cv2_bn)
        model_8_m_0_cv2_bn = None
        add_6_f = self.float_functional_simple_9.add(model_8_cv1_act, model_8_m_0_cv2_act)
        model_8_cv1_act = None
        model_8_m_0_cv2_act = None
        model_8_cv2_conv = self.model_8_cv2_conv(model_7_act)
        model_7_act = None
        model_8_cv2_bn = self.model_8_cv2_bn(model_8_cv2_conv)
        model_8_cv2_conv = None
        model_8_cv2_act = self.model_8_cv2_act(model_8_cv2_bn)
        model_8_cv2_bn = None
        cat_3_f = self.float_functional_simple_10.cat([add_6_f, model_8_cv2_act], 1)
        add_6_f = None
        model_8_cv2_act = None
        model_8_cv3_conv = self.model_8_cv3_conv(cat_3_f)
        cat_3_f = None
        model_8_cv3_bn = self.model_8_cv3_bn(model_8_cv3_conv)
        model_8_cv3_conv = None
        model_8_cv3_act = self.model_8_cv3_act(model_8_cv3_bn)
        model_8_cv3_bn = None
        model_9_cv1_conv = self.model_9_cv1_conv(model_8_cv3_act)
        model_8_cv3_act = None
        model_9_cv1_bn = self.model_9_cv1_bn(model_9_cv1_conv)
        model_9_cv1_conv = None
        model_9_cv1_act = self.model_9_cv1_act(model_9_cv1_bn)
        model_9_cv1_bn = None
        model_9_m = self.model_9_m(model_9_cv1_act)
        model_9_m_1 = self.model_9_m_1(model_9_m)
        model_9_m_2 = self.model_9_m_2(model_9_m_1)
        cat_4_f = self.float_functional_simple_11.cat([model_9_cv1_act, model_9_m, model_9_m_1, model_9_m_2], 1)
        model_9_cv1_act = None
        model_9_m = None
        model_9_m_1 = None
        model_9_m_2 = None
        model_9_cv2_conv = self.model_9_cv2_conv(cat_4_f)
        cat_4_f = None
        model_9_cv2_bn = self.model_9_cv2_bn(model_9_cv2_conv)
        model_9_cv2_conv = None
        model_9_cv2_act = self.model_9_cv2_act(model_9_cv2_bn)
        model_9_cv2_bn = None
        model_10_conv = self.model_10_conv(model_9_cv2_act)
        model_9_cv2_act = None
        model_10_bn = self.model_10_bn(model_10_conv)
        model_10_conv = None
        model_10_act = self.model_10_act(model_10_bn)
        model_10_bn = None
        model_11 = self.model_11(model_10_act)
        cat_5_f = self.float_functional_simple_12.cat([model_11, model_6_cv3_act], 1)
        model_11 = None
        model_6_cv3_act = None
        model_13_cv1_conv = self.model_13_cv1_conv(cat_5_f)
        model_13_cv1_bn = self.model_13_cv1_bn(model_13_cv1_conv)
        model_13_cv1_conv = None
        model_13_cv1_act = self.model_13_cv1_act(model_13_cv1_bn)
        model_13_cv1_bn = None
        model_13_m_0_cv1_conv = self.model_13_m_0_cv1_conv(model_13_cv1_act)
        model_13_cv1_act = None
        model_13_m_0_cv1_bn = self.model_13_m_0_cv1_bn(model_13_m_0_cv1_conv)
        model_13_m_0_cv1_conv = None
        model_13_m_0_cv1_act = self.model_13_m_0_cv1_act(model_13_m_0_cv1_bn)
        model_13_m_0_cv1_bn = None
        model_13_m_0_cv2_conv = self.model_13_m_0_cv2_conv(model_13_m_0_cv1_act)
        model_13_m_0_cv1_act = None
        model_13_m_0_cv2_bn = self.model_13_m_0_cv2_bn(model_13_m_0_cv2_conv)
        model_13_m_0_cv2_conv = None
        model_13_m_0_cv2_act = self.model_13_m_0_cv2_act(model_13_m_0_cv2_bn)
        model_13_m_0_cv2_bn = None
        model_13_cv2_conv = self.model_13_cv2_conv(cat_5_f)
        cat_5_f = None
        model_13_cv2_bn = self.model_13_cv2_bn(model_13_cv2_conv)
        model_13_cv2_conv = None
        model_13_cv2_act = self.model_13_cv2_act(model_13_cv2_bn)
        model_13_cv2_bn = None
        cat_6_f = self.float_functional_simple_13.cat([model_13_m_0_cv2_act, model_13_cv2_act], 1)
        model_13_m_0_cv2_act = None
        model_13_cv2_act = None
        model_13_cv3_conv = self.model_13_cv3_conv(cat_6_f)
        cat_6_f = None
        model_13_cv3_bn = self.model_13_cv3_bn(model_13_cv3_conv)
        model_13_cv3_conv = None
        model_13_cv3_act = self.model_13_cv3_act(model_13_cv3_bn)
        model_13_cv3_bn = None
        model_14_conv = self.model_14_conv(model_13_cv3_act)
        model_13_cv3_act = None
        model_14_bn = self.model_14_bn(model_14_conv)
        model_14_conv = None
        model_14_act = self.model_14_act(model_14_bn)
        model_14_bn = None
        model_15 = self.model_15(model_14_act)
        cat_7_f = self.float_functional_simple_14.cat([model_15, model_4_cv3_act], 1)
        model_15 = None
        model_4_cv3_act = None
        model_17_cv1_conv = self.model_17_cv1_conv(cat_7_f)
        model_17_cv1_bn = self.model_17_cv1_bn(model_17_cv1_conv)
        model_17_cv1_conv = None
        model_17_cv1_act = self.model_17_cv1_act(model_17_cv1_bn)
        model_17_cv1_bn = None
        model_17_m_0_cv1_conv = self.model_17_m_0_cv1_conv(model_17_cv1_act)
        model_17_cv1_act = None
        model_17_m_0_cv1_bn = self.model_17_m_0_cv1_bn(model_17_m_0_cv1_conv)
        model_17_m_0_cv1_conv = None
        model_17_m_0_cv1_act = self.model_17_m_0_cv1_act(model_17_m_0_cv1_bn)
        model_17_m_0_cv1_bn = None
        model_17_m_0_cv2_conv = self.model_17_m_0_cv2_conv(model_17_m_0_cv1_act)
        model_17_m_0_cv1_act = None
        model_17_m_0_cv2_bn = self.model_17_m_0_cv2_bn(model_17_m_0_cv2_conv)
        model_17_m_0_cv2_conv = None
        model_17_m_0_cv2_act = self.model_17_m_0_cv2_act(model_17_m_0_cv2_bn)
        model_17_m_0_cv2_bn = None
        model_17_cv2_conv = self.model_17_cv2_conv(cat_7_f)
        cat_7_f = None
        model_17_cv2_bn = self.model_17_cv2_bn(model_17_cv2_conv)
        model_17_cv2_conv = None
        model_17_cv2_act = self.model_17_cv2_act(model_17_cv2_bn)
        model_17_cv2_bn = None
        cat_8_f = self.float_functional_simple_15.cat([model_17_m_0_cv2_act, model_17_cv2_act], 1)
        model_17_m_0_cv2_act = None
        model_17_cv2_act = None
        model_17_cv3_conv = self.model_17_cv3_conv(cat_8_f)
        cat_8_f = None
        model_17_cv3_bn = self.model_17_cv3_bn(model_17_cv3_conv)
        model_17_cv3_conv = None
        model_17_cv3_act = self.model_17_cv3_act(model_17_cv3_bn)
        model_17_cv3_bn = None
        model_18_conv = self.model_18_conv(model_17_cv3_act)
        model_18_bn = self.model_18_bn(model_18_conv)
        model_18_conv = None
        model_18_act = self.model_18_act(model_18_bn)
        model_18_bn = None
        cat_9_f = self.float_functional_simple_16.cat([model_18_act, model_14_act], 1)
        model_18_act = None
        model_14_act = None
        model_20_cv1_conv = self.model_20_cv1_conv(cat_9_f)
        model_20_cv1_bn = self.model_20_cv1_bn(model_20_cv1_conv)
        model_20_cv1_conv = None
        model_20_cv1_act = self.model_20_cv1_act(model_20_cv1_bn)
        model_20_cv1_bn = None
        model_20_m_0_cv1_conv = self.model_20_m_0_cv1_conv(model_20_cv1_act)
        model_20_cv1_act = None
        model_20_m_0_cv1_bn = self.model_20_m_0_cv1_bn(model_20_m_0_cv1_conv)
        model_20_m_0_cv1_conv = None
        model_20_m_0_cv1_act = self.model_20_m_0_cv1_act(model_20_m_0_cv1_bn)
        model_20_m_0_cv1_bn = None
        model_20_m_0_cv2_conv = self.model_20_m_0_cv2_conv(model_20_m_0_cv1_act)
        model_20_m_0_cv1_act = None
        model_20_m_0_cv2_bn = self.model_20_m_0_cv2_bn(model_20_m_0_cv2_conv)
        model_20_m_0_cv2_conv = None
        model_20_m_0_cv2_act = self.model_20_m_0_cv2_act(model_20_m_0_cv2_bn)
        model_20_m_0_cv2_bn = None
        model_20_cv2_conv = self.model_20_cv2_conv(cat_9_f)
        cat_9_f = None
        model_20_cv2_bn = self.model_20_cv2_bn(model_20_cv2_conv)
        model_20_cv2_conv = None
        model_20_cv2_act = self.model_20_cv2_act(model_20_cv2_bn)
        model_20_cv2_bn = None
        cat_10_f = self.float_functional_simple_17.cat([model_20_m_0_cv2_act, model_20_cv2_act], 1)
        model_20_m_0_cv2_act = None
        model_20_cv2_act = None
        model_20_cv3_conv = self.model_20_cv3_conv(cat_10_f)
        cat_10_f = None
        model_20_cv3_bn = self.model_20_cv3_bn(model_20_cv3_conv)
        model_20_cv3_conv = None
        model_20_cv3_act = self.model_20_cv3_act(model_20_cv3_bn)
        model_20_cv3_bn = None
        model_21_conv = self.model_21_conv(model_20_cv3_act)
        model_21_bn = self.model_21_bn(model_21_conv)
        model_21_conv = None
        model_21_act = self.model_21_act(model_21_bn)
        model_21_bn = None
        cat_11_f = self.float_functional_simple_18.cat([model_21_act, model_10_act], 1)
        model_21_act = None
        model_10_act = None
        model_23_cv1_conv = self.model_23_cv1_conv(cat_11_f)
        model_23_cv1_bn = self.model_23_cv1_bn(model_23_cv1_conv)
        model_23_cv1_conv = None
        model_23_cv1_act = self.model_23_cv1_act(model_23_cv1_bn)
        model_23_cv1_bn = None
        model_23_m_0_cv1_conv = self.model_23_m_0_cv1_conv(model_23_cv1_act)
        model_23_cv1_act = None
        model_23_m_0_cv1_bn = self.model_23_m_0_cv1_bn(model_23_m_0_cv1_conv)
        model_23_m_0_cv1_conv = None
        model_23_m_0_cv1_act = self.model_23_m_0_cv1_act(model_23_m_0_cv1_bn)
        model_23_m_0_cv1_bn = None
        model_23_m_0_cv2_conv = self.model_23_m_0_cv2_conv(model_23_m_0_cv1_act)
        model_23_m_0_cv1_act = None
        model_23_m_0_cv2_bn = self.model_23_m_0_cv2_bn(model_23_m_0_cv2_conv)
        model_23_m_0_cv2_conv = None
        model_23_m_0_cv2_act = self.model_23_m_0_cv2_act(model_23_m_0_cv2_bn)
        model_23_m_0_cv2_bn = None
        model_23_cv2_conv = self.model_23_cv2_conv(cat_11_f)
        cat_11_f = None
        model_23_cv2_bn = self.model_23_cv2_bn(model_23_cv2_conv)
        model_23_cv2_conv = None
        model_23_cv2_act = self.model_23_cv2_act(model_23_cv2_bn)
        model_23_cv2_bn = None
        cat_12_f = self.float_functional_simple_19.cat([model_23_m_0_cv2_act, model_23_cv2_act], 1)
        model_23_m_0_cv2_act = None
        model_23_cv2_act = None
        model_23_cv3_conv = self.model_23_cv3_conv(cat_12_f)
        cat_12_f = None
        model_23_cv3_bn = self.model_23_cv3_bn(model_23_cv3_conv)
        model_23_cv3_conv = None
        model_23_cv3_act = self.model_23_cv3_act(model_23_cv3_bn)
        model_23_cv3_bn = None
        model_24_m_0 = self.model_24_m_0(model_17_cv3_act)
        model_17_cv3_act = None
        shape_0_f = model_24_m_0.shape
        view_0_f = model_24_m_0.view(shape_0_f[0], 3, 85, shape_0_f[2], shape_0_f[3])
        model_24_m_0 = None
        shape_0_f = None
        permute_0_f = view_0_f.permute(0, 1, 3, 4, 2)
        view_0_f = None
        contiguous_0_f = permute_0_f.contiguous()
        permute_0_f = None
        model_24_m_1 = self.model_24_m_1(model_20_cv3_act)
        model_20_cv3_act = None
        shape_1_f = model_24_m_1.shape
        view_1_f = model_24_m_1.view(shape_1_f[0], 3, 85, shape_1_f[2], shape_1_f[3])
        model_24_m_1 = None
        shape_1_f = None
        permute_1_f = view_1_f.permute(0, 1, 3, 4, 2)
        view_1_f = None
        contiguous_1_f = permute_1_f.contiguous()
        permute_1_f = None
        model_24_m_2 = self.model_24_m_2(model_23_cv3_act)
        model_23_cv3_act = None
        shape_2_f = model_24_m_2.shape
        view_2_f = model_24_m_2.view(shape_2_f[0], 3, 85, shape_2_f[2], shape_2_f[3])
        model_24_m_2 = None
        shape_2_f = None
        permute_2_f = view_2_f.permute(0, 1, 3, 4, 2)
        view_2_f = None
        contiguous_2_f = permute_2_f.contiguous()
        permute_2_f = None
        fake_dequant_0 = self.fake_dequant_0(contiguous_0_f)
        contiguous_0_f = None
        fake_dequant_1 = self.fake_dequant_1(contiguous_1_f)
        contiguous_1_f = None
        fake_dequant_2 = self.fake_dequant_2(contiguous_2_f)
        contiguous_2_f = None
        return fake_dequant_0, fake_dequant_1, fake_dequant_2


if __name__ == "__main__":
    model = Model_qat()
    model.load_state_dict(torch.load('./quantization/model_qat.pth'))

    model.eval()
    model.cpu()

    dummy_input_0 = torch.ones((1, 3, 640, 640), dtype=torch.float32)

    output = model(dummy_input_0)
    print(output)

