
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
        self.fake_dequant_inner_0_0 = torch.quantization.DeQuantStub()
        self.model_0_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_56_0 = torch.quantization.QuantStub()
        self.model_1_conv = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_1_0 = torch.quantization.DeQuantStub()
        self.model_1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_55_0 = torch.quantization.QuantStub()
        self.model_2_cv1_conv = torch.nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_2_cv1_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_2_0 = torch.quantization.DeQuantStub()
        self.model_2_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_54_0 = torch.quantization.QuantStub()
        self.model_2_m_0_cv1_conv = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_2_m_0_cv1_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_3_0 = torch.quantization.DeQuantStub()
        self.model_2_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_53_0 = torch.quantization.QuantStub()
        self.model_2_m_0_cv2_conv = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_2_m_0_cv2_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_4_0 = torch.quantization.DeQuantStub()
        self.model_2_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_52_0 = torch.quantization.QuantStub()
        self.model_2_cv2_conv = torch.nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_2_cv2_bn = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_5_0 = torch.quantization.DeQuantStub()
        self.model_2_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_51_0 = torch.quantization.QuantStub()
        self.model_2_cv3_conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_2_cv3_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_6_0 = torch.quantization.DeQuantStub()
        self.model_2_cv3_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_50_0 = torch.quantization.QuantStub()
        self.model_3_conv = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_3_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_7_0 = torch.quantization.DeQuantStub()
        self.model_3_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_49_0 = torch.quantization.QuantStub()
        self.model_4_cv1_conv = torch.nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_8_0 = torch.quantization.DeQuantStub()
        self.model_4_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_48_0 = torch.quantization.QuantStub()
        self.model_4_m_0_cv1_conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_m_0_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_9_0 = torch.quantization.DeQuantStub()
        self.model_4_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_47_0 = torch.quantization.QuantStub()
        self.model_4_m_0_cv2_conv = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_4_m_0_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_10_0 = torch.quantization.DeQuantStub()
        self.model_4_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_46_0 = torch.quantization.QuantStub()
        self.model_4_m_1_cv1_conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_m_1_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_11_0 = torch.quantization.DeQuantStub()
        self.model_4_m_1_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_45_0 = torch.quantization.QuantStub()
        self.model_4_m_1_cv2_conv = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_4_m_1_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_12_0 = torch.quantization.DeQuantStub()
        self.model_4_m_1_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_44_0 = torch.quantization.QuantStub()
        self.model_4_cv2_conv = torch.nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_13_0 = torch.quantization.DeQuantStub()
        self.model_4_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_43_0 = torch.quantization.QuantStub()
        self.model_4_cv3_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_4_cv3_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_14_0 = torch.quantization.DeQuantStub()
        self.model_4_cv3_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_42_0 = torch.quantization.QuantStub()
        self.model_5_conv = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_5_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_15_0 = torch.quantization.DeQuantStub()
        self.model_5_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_41_0 = torch.quantization.QuantStub()
        self.model_6_cv1_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_16_0 = torch.quantization.DeQuantStub()
        self.model_6_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_40_0 = torch.quantization.QuantStub()
        self.model_6_m_0_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_m_0_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_17_0 = torch.quantization.DeQuantStub()
        self.model_6_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_39_0 = torch.quantization.QuantStub()
        self.model_6_m_0_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_6_m_0_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_18_0 = torch.quantization.DeQuantStub()
        self.model_6_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_38_0 = torch.quantization.QuantStub()
        self.model_6_m_1_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_m_1_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_19_0 = torch.quantization.DeQuantStub()
        self.model_6_m_1_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_37_0 = torch.quantization.QuantStub()
        self.model_6_m_1_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_6_m_1_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_20_0 = torch.quantization.DeQuantStub()
        self.model_6_m_1_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_36_0 = torch.quantization.QuantStub()
        self.model_6_m_2_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_m_2_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_21_0 = torch.quantization.DeQuantStub()
        self.model_6_m_2_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_35_0 = torch.quantization.QuantStub()
        self.model_6_m_2_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_6_m_2_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_22_0 = torch.quantization.DeQuantStub()
        self.model_6_m_2_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_34_0 = torch.quantization.QuantStub()
        self.model_6_cv2_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_23_0 = torch.quantization.DeQuantStub()
        self.model_6_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_33_0 = torch.quantization.QuantStub()
        self.model_6_cv3_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_6_cv3_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_24_0 = torch.quantization.DeQuantStub()
        self.model_6_cv3_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_32_0 = torch.quantization.QuantStub()
        self.model_7_conv = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_7_bn = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_25_0 = torch.quantization.DeQuantStub()
        self.model_7_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_31_0 = torch.quantization.QuantStub()
        self.model_8_cv1_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_8_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_26_0 = torch.quantization.DeQuantStub()
        self.model_8_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_30_0 = torch.quantization.QuantStub()
        self.model_8_m_0_cv1_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_8_m_0_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_27_0 = torch.quantization.DeQuantStub()
        self.model_8_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_29_0 = torch.quantization.QuantStub()
        self.model_8_m_0_cv2_conv = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_8_m_0_cv2_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_28_0 = torch.quantization.DeQuantStub()
        self.model_8_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_28_0 = torch.quantization.QuantStub()
        self.model_8_cv2_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_8_cv2_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_29_0 = torch.quantization.DeQuantStub()
        self.model_8_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_27_0 = torch.quantization.QuantStub()
        self.model_8_cv3_conv = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_8_cv3_bn = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_30_0 = torch.quantization.DeQuantStub()
        self.model_8_cv3_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_26_0 = torch.quantization.QuantStub()
        self.model_9_cv1_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_9_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_31_0 = torch.quantization.DeQuantStub()
        self.model_9_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_25_0 = torch.quantization.QuantStub()
        self.model_9_m = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.model_9_m_1 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.model_9_m_2 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.model_9_cv2_conv = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_9_cv2_bn = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_32_0 = torch.quantization.DeQuantStub()
        self.model_9_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_24_0 = torch.quantization.QuantStub()
        self.model_10_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_10_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_33_0 = torch.quantization.DeQuantStub()
        self.model_10_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_23_0 = torch.quantization.QuantStub()
        self.model_11 = torch.nn.Upsample(scale_factor=2.0)
        self.model_13_cv1_conv = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_13_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_34_0 = torch.quantization.DeQuantStub()
        self.model_13_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_22_0 = torch.quantization.QuantStub()
        self.model_13_m_0_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_13_m_0_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_35_0 = torch.quantization.DeQuantStub()
        self.model_13_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_21_0 = torch.quantization.QuantStub()
        self.model_13_m_0_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_13_m_0_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_36_0 = torch.quantization.DeQuantStub()
        self.model_13_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_13_cv2_conv = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_13_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_37_0 = torch.quantization.DeQuantStub()
        self.model_13_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_19_0 = torch.quantization.QuantStub()
        self.fake_quant_inner_20_0 = torch.quantization.QuantStub()
        self.model_13_cv3_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_13_cv3_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_38_0 = torch.quantization.DeQuantStub()
        self.model_13_cv3_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_18_0 = torch.quantization.QuantStub()
        self.model_14_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_14_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_39_0 = torch.quantization.DeQuantStub()
        self.model_14_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_17_0 = torch.quantization.QuantStub()
        self.model_15 = torch.nn.Upsample(scale_factor=2.0)
        self.model_17_cv1_conv = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_17_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_40_0 = torch.quantization.DeQuantStub()
        self.model_17_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_16_0 = torch.quantization.QuantStub()
        self.model_17_m_0_cv1_conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_17_m_0_cv1_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_41_0 = torch.quantization.DeQuantStub()
        self.model_17_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_15_0 = torch.quantization.QuantStub()
        self.model_17_m_0_cv2_conv = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_17_m_0_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_42_0 = torch.quantization.DeQuantStub()
        self.model_17_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_17_cv2_conv = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_17_cv2_bn = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_43_0 = torch.quantization.DeQuantStub()
        self.model_17_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_13_0 = torch.quantization.QuantStub()
        self.fake_quant_inner_14_0 = torch.quantization.QuantStub()
        self.model_17_cv3_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_17_cv3_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_44_0 = torch.quantization.DeQuantStub()
        self.model_17_cv3_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_12_0 = torch.quantization.QuantStub()
        self.model_18_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_18_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_45_0 = torch.quantization.DeQuantStub()
        self.model_18_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_11_0 = torch.quantization.QuantStub()
        self.model_20_cv1_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_20_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_46_0 = torch.quantization.DeQuantStub()
        self.model_20_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_10_0 = torch.quantization.QuantStub()
        self.model_20_m_0_cv1_conv = torch.nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_20_m_0_cv1_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_47_0 = torch.quantization.DeQuantStub()
        self.model_20_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_9_0 = torch.quantization.QuantStub()
        self.model_20_m_0_cv2_conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_20_m_0_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_48_0 = torch.quantization.DeQuantStub()
        self.model_20_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_20_cv2_conv = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_20_cv2_bn = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_49_0 = torch.quantization.DeQuantStub()
        self.model_20_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_7_0 = torch.quantization.QuantStub()
        self.fake_quant_inner_8_0 = torch.quantization.QuantStub()
        self.model_20_cv3_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_20_cv3_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_50_0 = torch.quantization.DeQuantStub()
        self.model_20_cv3_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_6_0 = torch.quantization.QuantStub()
        self.model_21_conv = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model_21_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_51_0 = torch.quantization.DeQuantStub()
        self.model_21_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_5_0 = torch.quantization.QuantStub()
        self.model_23_cv1_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_23_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_52_0 = torch.quantization.DeQuantStub()
        self.model_23_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_4_0 = torch.quantization.QuantStub()
        self.model_23_m_0_cv1_conv = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_23_m_0_cv1_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_53_0 = torch.quantization.DeQuantStub()
        self.model_23_m_0_cv1_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_3_0 = torch.quantization.QuantStub()
        self.model_23_m_0_cv2_conv = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_23_m_0_cv2_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_54_0 = torch.quantization.DeQuantStub()
        self.model_23_m_0_cv2_act = torch.nn.SiLU(inplace=True)
        self.model_23_cv2_conv = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_23_cv2_bn = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_55_0 = torch.quantization.DeQuantStub()
        self.model_23_cv2_act = torch.nn.SiLU(inplace=True)
        self.fake_quant_inner_1_0 = torch.quantization.QuantStub()
        self.fake_quant_inner_2_0 = torch.quantization.QuantStub()
        self.model_23_cv3_conv = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model_23_cv3_bn = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.fake_dequant_inner_56_0 = torch.quantization.DeQuantStub()
        self.model_23_cv3_act = torch.nn.SiLU(inplace=True)
        self.model_24_m_0 = torch.nn.Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        self.model_24_m_1 = torch.nn.Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        self.fake_quant_inner_0_0 = torch.quantization.QuantStub()
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
        fake_dequant_inner_0_0 = self.fake_dequant_inner_0_0(model_0_bn)
        model_0_bn = None
        model_0_act = self.model_0_act(fake_dequant_inner_0_0)
        fake_dequant_inner_0_0 = None
        fake_quant_inner_56_0 = self.fake_quant_inner_56_0(model_0_act)
        model_0_act = None
        model_1_conv = self.model_1_conv(fake_quant_inner_56_0)
        fake_quant_inner_56_0 = None
        model_1_bn = self.model_1_bn(model_1_conv)
        model_1_conv = None
        fake_dequant_inner_1_0 = self.fake_dequant_inner_1_0(model_1_bn)
        model_1_bn = None
        model_1_act = self.model_1_act(fake_dequant_inner_1_0)
        fake_dequant_inner_1_0 = None
        fake_quant_inner_55_0 = self.fake_quant_inner_55_0(model_1_act)
        model_1_act = None
        model_2_cv1_conv = self.model_2_cv1_conv(fake_quant_inner_55_0)
        model_2_cv1_bn = self.model_2_cv1_bn(model_2_cv1_conv)
        model_2_cv1_conv = None
        fake_dequant_inner_2_0 = self.fake_dequant_inner_2_0(model_2_cv1_bn)
        model_2_cv1_bn = None
        model_2_cv1_act = self.model_2_cv1_act(fake_dequant_inner_2_0)
        fake_dequant_inner_2_0 = None
        fake_quant_inner_54_0 = self.fake_quant_inner_54_0(model_2_cv1_act)
        model_2_cv1_act = None
        model_2_m_0_cv1_conv = self.model_2_m_0_cv1_conv(fake_quant_inner_54_0)
        model_2_m_0_cv1_bn = self.model_2_m_0_cv1_bn(model_2_m_0_cv1_conv)
        model_2_m_0_cv1_conv = None
        fake_dequant_inner_3_0 = self.fake_dequant_inner_3_0(model_2_m_0_cv1_bn)
        model_2_m_0_cv1_bn = None
        model_2_m_0_cv1_act = self.model_2_m_0_cv1_act(fake_dequant_inner_3_0)
        fake_dequant_inner_3_0 = None
        fake_quant_inner_53_0 = self.fake_quant_inner_53_0(model_2_m_0_cv1_act)
        model_2_m_0_cv1_act = None
        model_2_m_0_cv2_conv = self.model_2_m_0_cv2_conv(fake_quant_inner_53_0)
        fake_quant_inner_53_0 = None
        model_2_m_0_cv2_bn = self.model_2_m_0_cv2_bn(model_2_m_0_cv2_conv)
        model_2_m_0_cv2_conv = None
        fake_dequant_inner_4_0 = self.fake_dequant_inner_4_0(model_2_m_0_cv2_bn)
        model_2_m_0_cv2_bn = None
        model_2_m_0_cv2_act = self.model_2_m_0_cv2_act(fake_dequant_inner_4_0)
        fake_dequant_inner_4_0 = None
        fake_quant_inner_52_0 = self.fake_quant_inner_52_0(model_2_m_0_cv2_act)
        model_2_m_0_cv2_act = None
        add_0_f = self.float_functional_simple_0.add(fake_quant_inner_54_0, fake_quant_inner_52_0)
        fake_quant_inner_54_0 = None
        fake_quant_inner_52_0 = None
        model_2_cv2_conv = self.model_2_cv2_conv(fake_quant_inner_55_0)
        fake_quant_inner_55_0 = None
        model_2_cv2_bn = self.model_2_cv2_bn(model_2_cv2_conv)
        model_2_cv2_conv = None
        fake_dequant_inner_5_0 = self.fake_dequant_inner_5_0(model_2_cv2_bn)
        model_2_cv2_bn = None
        model_2_cv2_act = self.model_2_cv2_act(fake_dequant_inner_5_0)
        fake_dequant_inner_5_0 = None
        fake_quant_inner_51_0 = self.fake_quant_inner_51_0(model_2_cv2_act)
        model_2_cv2_act = None
        cat_0_f = self.float_functional_simple_1.cat([add_0_f, fake_quant_inner_51_0], 1)
        add_0_f = None
        fake_quant_inner_51_0 = None
        model_2_cv3_conv = self.model_2_cv3_conv(cat_0_f)
        cat_0_f = None
        model_2_cv3_bn = self.model_2_cv3_bn(model_2_cv3_conv)
        model_2_cv3_conv = None
        fake_dequant_inner_6_0 = self.fake_dequant_inner_6_0(model_2_cv3_bn)
        model_2_cv3_bn = None
        model_2_cv3_act = self.model_2_cv3_act(fake_dequant_inner_6_0)
        fake_dequant_inner_6_0 = None
        fake_quant_inner_50_0 = self.fake_quant_inner_50_0(model_2_cv3_act)
        model_2_cv3_act = None
        model_3_conv = self.model_3_conv(fake_quant_inner_50_0)
        fake_quant_inner_50_0 = None
        model_3_bn = self.model_3_bn(model_3_conv)
        model_3_conv = None
        fake_dequant_inner_7_0 = self.fake_dequant_inner_7_0(model_3_bn)
        model_3_bn = None
        model_3_act = self.model_3_act(fake_dequant_inner_7_0)
        fake_dequant_inner_7_0 = None
        fake_quant_inner_49_0 = self.fake_quant_inner_49_0(model_3_act)
        model_3_act = None
        model_4_cv1_conv = self.model_4_cv1_conv(fake_quant_inner_49_0)
        model_4_cv1_bn = self.model_4_cv1_bn(model_4_cv1_conv)
        model_4_cv1_conv = None
        fake_dequant_inner_8_0 = self.fake_dequant_inner_8_0(model_4_cv1_bn)
        model_4_cv1_bn = None
        model_4_cv1_act = self.model_4_cv1_act(fake_dequant_inner_8_0)
        fake_dequant_inner_8_0 = None
        fake_quant_inner_48_0 = self.fake_quant_inner_48_0(model_4_cv1_act)
        model_4_cv1_act = None
        model_4_m_0_cv1_conv = self.model_4_m_0_cv1_conv(fake_quant_inner_48_0)
        model_4_m_0_cv1_bn = self.model_4_m_0_cv1_bn(model_4_m_0_cv1_conv)
        model_4_m_0_cv1_conv = None
        fake_dequant_inner_9_0 = self.fake_dequant_inner_9_0(model_4_m_0_cv1_bn)
        model_4_m_0_cv1_bn = None
        model_4_m_0_cv1_act = self.model_4_m_0_cv1_act(fake_dequant_inner_9_0)
        fake_dequant_inner_9_0 = None
        fake_quant_inner_47_0 = self.fake_quant_inner_47_0(model_4_m_0_cv1_act)
        model_4_m_0_cv1_act = None
        model_4_m_0_cv2_conv = self.model_4_m_0_cv2_conv(fake_quant_inner_47_0)
        fake_quant_inner_47_0 = None
        model_4_m_0_cv2_bn = self.model_4_m_0_cv2_bn(model_4_m_0_cv2_conv)
        model_4_m_0_cv2_conv = None
        fake_dequant_inner_10_0 = self.fake_dequant_inner_10_0(model_4_m_0_cv2_bn)
        model_4_m_0_cv2_bn = None
        model_4_m_0_cv2_act = self.model_4_m_0_cv2_act(fake_dequant_inner_10_0)
        fake_dequant_inner_10_0 = None
        fake_quant_inner_46_0 = self.fake_quant_inner_46_0(model_4_m_0_cv2_act)
        model_4_m_0_cv2_act = None
        add_1_f = self.float_functional_simple_2.add(fake_quant_inner_48_0, fake_quant_inner_46_0)
        fake_quant_inner_48_0 = None
        fake_quant_inner_46_0 = None
        model_4_m_1_cv1_conv = self.model_4_m_1_cv1_conv(add_1_f)
        model_4_m_1_cv1_bn = self.model_4_m_1_cv1_bn(model_4_m_1_cv1_conv)
        model_4_m_1_cv1_conv = None
        fake_dequant_inner_11_0 = self.fake_dequant_inner_11_0(model_4_m_1_cv1_bn)
        model_4_m_1_cv1_bn = None
        model_4_m_1_cv1_act = self.model_4_m_1_cv1_act(fake_dequant_inner_11_0)
        fake_dequant_inner_11_0 = None
        fake_quant_inner_45_0 = self.fake_quant_inner_45_0(model_4_m_1_cv1_act)
        model_4_m_1_cv1_act = None
        model_4_m_1_cv2_conv = self.model_4_m_1_cv2_conv(fake_quant_inner_45_0)
        fake_quant_inner_45_0 = None
        model_4_m_1_cv2_bn = self.model_4_m_1_cv2_bn(model_4_m_1_cv2_conv)
        model_4_m_1_cv2_conv = None
        fake_dequant_inner_12_0 = self.fake_dequant_inner_12_0(model_4_m_1_cv2_bn)
        model_4_m_1_cv2_bn = None
        model_4_m_1_cv2_act = self.model_4_m_1_cv2_act(fake_dequant_inner_12_0)
        fake_dequant_inner_12_0 = None
        fake_quant_inner_44_0 = self.fake_quant_inner_44_0(model_4_m_1_cv2_act)
        model_4_m_1_cv2_act = None
        add_2_f = self.float_functional_simple_3.add(add_1_f, fake_quant_inner_44_0)
        add_1_f = None
        fake_quant_inner_44_0 = None
        model_4_cv2_conv = self.model_4_cv2_conv(fake_quant_inner_49_0)
        fake_quant_inner_49_0 = None
        model_4_cv2_bn = self.model_4_cv2_bn(model_4_cv2_conv)
        model_4_cv2_conv = None
        fake_dequant_inner_13_0 = self.fake_dequant_inner_13_0(model_4_cv2_bn)
        model_4_cv2_bn = None
        model_4_cv2_act = self.model_4_cv2_act(fake_dequant_inner_13_0)
        fake_dequant_inner_13_0 = None
        fake_quant_inner_43_0 = self.fake_quant_inner_43_0(model_4_cv2_act)
        model_4_cv2_act = None
        cat_1_f = self.float_functional_simple_4.cat([add_2_f, fake_quant_inner_43_0], 1)
        add_2_f = None
        fake_quant_inner_43_0 = None
        model_4_cv3_conv = self.model_4_cv3_conv(cat_1_f)
        cat_1_f = None
        model_4_cv3_bn = self.model_4_cv3_bn(model_4_cv3_conv)
        model_4_cv3_conv = None
        fake_dequant_inner_14_0 = self.fake_dequant_inner_14_0(model_4_cv3_bn)
        model_4_cv3_bn = None
        model_4_cv3_act = self.model_4_cv3_act(fake_dequant_inner_14_0)
        fake_dequant_inner_14_0 = None
        fake_quant_inner_42_0 = self.fake_quant_inner_42_0(model_4_cv3_act)
        model_4_cv3_act = None
        model_5_conv = self.model_5_conv(fake_quant_inner_42_0)
        model_5_bn = self.model_5_bn(model_5_conv)
        model_5_conv = None
        fake_dequant_inner_15_0 = self.fake_dequant_inner_15_0(model_5_bn)
        model_5_bn = None
        model_5_act = self.model_5_act(fake_dequant_inner_15_0)
        fake_dequant_inner_15_0 = None
        fake_quant_inner_41_0 = self.fake_quant_inner_41_0(model_5_act)
        model_5_act = None
        model_6_cv1_conv = self.model_6_cv1_conv(fake_quant_inner_41_0)
        model_6_cv1_bn = self.model_6_cv1_bn(model_6_cv1_conv)
        model_6_cv1_conv = None
        fake_dequant_inner_16_0 = self.fake_dequant_inner_16_0(model_6_cv1_bn)
        model_6_cv1_bn = None
        model_6_cv1_act = self.model_6_cv1_act(fake_dequant_inner_16_0)
        fake_dequant_inner_16_0 = None
        fake_quant_inner_40_0 = self.fake_quant_inner_40_0(model_6_cv1_act)
        model_6_cv1_act = None
        model_6_m_0_cv1_conv = self.model_6_m_0_cv1_conv(fake_quant_inner_40_0)
        model_6_m_0_cv1_bn = self.model_6_m_0_cv1_bn(model_6_m_0_cv1_conv)
        model_6_m_0_cv1_conv = None
        fake_dequant_inner_17_0 = self.fake_dequant_inner_17_0(model_6_m_0_cv1_bn)
        model_6_m_0_cv1_bn = None
        model_6_m_0_cv1_act = self.model_6_m_0_cv1_act(fake_dequant_inner_17_0)
        fake_dequant_inner_17_0 = None
        fake_quant_inner_39_0 = self.fake_quant_inner_39_0(model_6_m_0_cv1_act)
        model_6_m_0_cv1_act = None
        model_6_m_0_cv2_conv = self.model_6_m_0_cv2_conv(fake_quant_inner_39_0)
        fake_quant_inner_39_0 = None
        model_6_m_0_cv2_bn = self.model_6_m_0_cv2_bn(model_6_m_0_cv2_conv)
        model_6_m_0_cv2_conv = None
        fake_dequant_inner_18_0 = self.fake_dequant_inner_18_0(model_6_m_0_cv2_bn)
        model_6_m_0_cv2_bn = None
        model_6_m_0_cv2_act = self.model_6_m_0_cv2_act(fake_dequant_inner_18_0)
        fake_dequant_inner_18_0 = None
        fake_quant_inner_38_0 = self.fake_quant_inner_38_0(model_6_m_0_cv2_act)
        model_6_m_0_cv2_act = None
        add_3_f = self.float_functional_simple_5.add(fake_quant_inner_40_0, fake_quant_inner_38_0)
        fake_quant_inner_40_0 = None
        fake_quant_inner_38_0 = None
        model_6_m_1_cv1_conv = self.model_6_m_1_cv1_conv(add_3_f)
        model_6_m_1_cv1_bn = self.model_6_m_1_cv1_bn(model_6_m_1_cv1_conv)
        model_6_m_1_cv1_conv = None
        fake_dequant_inner_19_0 = self.fake_dequant_inner_19_0(model_6_m_1_cv1_bn)
        model_6_m_1_cv1_bn = None
        model_6_m_1_cv1_act = self.model_6_m_1_cv1_act(fake_dequant_inner_19_0)
        fake_dequant_inner_19_0 = None
        fake_quant_inner_37_0 = self.fake_quant_inner_37_0(model_6_m_1_cv1_act)
        model_6_m_1_cv1_act = None
        model_6_m_1_cv2_conv = self.model_6_m_1_cv2_conv(fake_quant_inner_37_0)
        fake_quant_inner_37_0 = None
        model_6_m_1_cv2_bn = self.model_6_m_1_cv2_bn(model_6_m_1_cv2_conv)
        model_6_m_1_cv2_conv = None
        fake_dequant_inner_20_0 = self.fake_dequant_inner_20_0(model_6_m_1_cv2_bn)
        model_6_m_1_cv2_bn = None
        model_6_m_1_cv2_act = self.model_6_m_1_cv2_act(fake_dequant_inner_20_0)
        fake_dequant_inner_20_0 = None
        fake_quant_inner_36_0 = self.fake_quant_inner_36_0(model_6_m_1_cv2_act)
        model_6_m_1_cv2_act = None
        add_4_f = self.float_functional_simple_6.add(add_3_f, fake_quant_inner_36_0)
        add_3_f = None
        fake_quant_inner_36_0 = None
        model_6_m_2_cv1_conv = self.model_6_m_2_cv1_conv(add_4_f)
        model_6_m_2_cv1_bn = self.model_6_m_2_cv1_bn(model_6_m_2_cv1_conv)
        model_6_m_2_cv1_conv = None
        fake_dequant_inner_21_0 = self.fake_dequant_inner_21_0(model_6_m_2_cv1_bn)
        model_6_m_2_cv1_bn = None
        model_6_m_2_cv1_act = self.model_6_m_2_cv1_act(fake_dequant_inner_21_0)
        fake_dequant_inner_21_0 = None
        fake_quant_inner_35_0 = self.fake_quant_inner_35_0(model_6_m_2_cv1_act)
        model_6_m_2_cv1_act = None
        model_6_m_2_cv2_conv = self.model_6_m_2_cv2_conv(fake_quant_inner_35_0)
        fake_quant_inner_35_0 = None
        model_6_m_2_cv2_bn = self.model_6_m_2_cv2_bn(model_6_m_2_cv2_conv)
        model_6_m_2_cv2_conv = None
        fake_dequant_inner_22_0 = self.fake_dequant_inner_22_0(model_6_m_2_cv2_bn)
        model_6_m_2_cv2_bn = None
        model_6_m_2_cv2_act = self.model_6_m_2_cv2_act(fake_dequant_inner_22_0)
        fake_dequant_inner_22_0 = None
        fake_quant_inner_34_0 = self.fake_quant_inner_34_0(model_6_m_2_cv2_act)
        model_6_m_2_cv2_act = None
        add_5_f = self.float_functional_simple_7.add(add_4_f, fake_quant_inner_34_0)
        add_4_f = None
        fake_quant_inner_34_0 = None
        model_6_cv2_conv = self.model_6_cv2_conv(fake_quant_inner_41_0)
        fake_quant_inner_41_0 = None
        model_6_cv2_bn = self.model_6_cv2_bn(model_6_cv2_conv)
        model_6_cv2_conv = None
        fake_dequant_inner_23_0 = self.fake_dequant_inner_23_0(model_6_cv2_bn)
        model_6_cv2_bn = None
        model_6_cv2_act = self.model_6_cv2_act(fake_dequant_inner_23_0)
        fake_dequant_inner_23_0 = None
        fake_quant_inner_33_0 = self.fake_quant_inner_33_0(model_6_cv2_act)
        model_6_cv2_act = None
        cat_2_f = self.float_functional_simple_8.cat([add_5_f, fake_quant_inner_33_0], 1)
        add_5_f = None
        fake_quant_inner_33_0 = None
        model_6_cv3_conv = self.model_6_cv3_conv(cat_2_f)
        cat_2_f = None
        model_6_cv3_bn = self.model_6_cv3_bn(model_6_cv3_conv)
        model_6_cv3_conv = None
        fake_dequant_inner_24_0 = self.fake_dequant_inner_24_0(model_6_cv3_bn)
        model_6_cv3_bn = None
        model_6_cv3_act = self.model_6_cv3_act(fake_dequant_inner_24_0)
        fake_dequant_inner_24_0 = None
        fake_quant_inner_32_0 = self.fake_quant_inner_32_0(model_6_cv3_act)
        model_6_cv3_act = None
        model_7_conv = self.model_7_conv(fake_quant_inner_32_0)
        model_7_bn = self.model_7_bn(model_7_conv)
        model_7_conv = None
        fake_dequant_inner_25_0 = self.fake_dequant_inner_25_0(model_7_bn)
        model_7_bn = None
        model_7_act = self.model_7_act(fake_dequant_inner_25_0)
        fake_dequant_inner_25_0 = None
        fake_quant_inner_31_0 = self.fake_quant_inner_31_0(model_7_act)
        model_7_act = None
        model_8_cv1_conv = self.model_8_cv1_conv(fake_quant_inner_31_0)
        model_8_cv1_bn = self.model_8_cv1_bn(model_8_cv1_conv)
        model_8_cv1_conv = None
        fake_dequant_inner_26_0 = self.fake_dequant_inner_26_0(model_8_cv1_bn)
        model_8_cv1_bn = None
        model_8_cv1_act = self.model_8_cv1_act(fake_dequant_inner_26_0)
        fake_dequant_inner_26_0 = None
        fake_quant_inner_30_0 = self.fake_quant_inner_30_0(model_8_cv1_act)
        model_8_cv1_act = None
        model_8_m_0_cv1_conv = self.model_8_m_0_cv1_conv(fake_quant_inner_30_0)
        model_8_m_0_cv1_bn = self.model_8_m_0_cv1_bn(model_8_m_0_cv1_conv)
        model_8_m_0_cv1_conv = None
        fake_dequant_inner_27_0 = self.fake_dequant_inner_27_0(model_8_m_0_cv1_bn)
        model_8_m_0_cv1_bn = None
        model_8_m_0_cv1_act = self.model_8_m_0_cv1_act(fake_dequant_inner_27_0)
        fake_dequant_inner_27_0 = None
        fake_quant_inner_29_0 = self.fake_quant_inner_29_0(model_8_m_0_cv1_act)
        model_8_m_0_cv1_act = None
        model_8_m_0_cv2_conv = self.model_8_m_0_cv2_conv(fake_quant_inner_29_0)
        fake_quant_inner_29_0 = None
        model_8_m_0_cv2_bn = self.model_8_m_0_cv2_bn(model_8_m_0_cv2_conv)
        model_8_m_0_cv2_conv = None
        fake_dequant_inner_28_0 = self.fake_dequant_inner_28_0(model_8_m_0_cv2_bn)
        model_8_m_0_cv2_bn = None
        model_8_m_0_cv2_act = self.model_8_m_0_cv2_act(fake_dequant_inner_28_0)
        fake_dequant_inner_28_0 = None
        fake_quant_inner_28_0 = self.fake_quant_inner_28_0(model_8_m_0_cv2_act)
        model_8_m_0_cv2_act = None
        add_6_f = self.float_functional_simple_9.add(fake_quant_inner_30_0, fake_quant_inner_28_0)
        fake_quant_inner_30_0 = None
        fake_quant_inner_28_0 = None
        model_8_cv2_conv = self.model_8_cv2_conv(fake_quant_inner_31_0)
        fake_quant_inner_31_0 = None
        model_8_cv2_bn = self.model_8_cv2_bn(model_8_cv2_conv)
        model_8_cv2_conv = None
        fake_dequant_inner_29_0 = self.fake_dequant_inner_29_0(model_8_cv2_bn)
        model_8_cv2_bn = None
        model_8_cv2_act = self.model_8_cv2_act(fake_dequant_inner_29_0)
        fake_dequant_inner_29_0 = None
        fake_quant_inner_27_0 = self.fake_quant_inner_27_0(model_8_cv2_act)
        model_8_cv2_act = None
        cat_3_f = self.float_functional_simple_10.cat([add_6_f, fake_quant_inner_27_0], 1)
        add_6_f = None
        fake_quant_inner_27_0 = None
        model_8_cv3_conv = self.model_8_cv3_conv(cat_3_f)
        cat_3_f = None
        model_8_cv3_bn = self.model_8_cv3_bn(model_8_cv3_conv)
        model_8_cv3_conv = None
        fake_dequant_inner_30_0 = self.fake_dequant_inner_30_0(model_8_cv3_bn)
        model_8_cv3_bn = None
        model_8_cv3_act = self.model_8_cv3_act(fake_dequant_inner_30_0)
        fake_dequant_inner_30_0 = None
        fake_quant_inner_26_0 = self.fake_quant_inner_26_0(model_8_cv3_act)
        model_8_cv3_act = None
        model_9_cv1_conv = self.model_9_cv1_conv(fake_quant_inner_26_0)
        fake_quant_inner_26_0 = None
        model_9_cv1_bn = self.model_9_cv1_bn(model_9_cv1_conv)
        model_9_cv1_conv = None
        fake_dequant_inner_31_0 = self.fake_dequant_inner_31_0(model_9_cv1_bn)
        model_9_cv1_bn = None
        model_9_cv1_act = self.model_9_cv1_act(fake_dequant_inner_31_0)
        fake_dequant_inner_31_0 = None
        fake_quant_inner_25_0 = self.fake_quant_inner_25_0(model_9_cv1_act)
        model_9_cv1_act = None
        model_9_m = self.model_9_m(fake_quant_inner_25_0)
        model_9_m_1 = self.model_9_m_1(model_9_m)
        model_9_m_2 = self.model_9_m_2(model_9_m_1)
        cat_4_f = self.float_functional_simple_11.cat([fake_quant_inner_25_0, model_9_m, model_9_m_1, model_9_m_2], 1)
        fake_quant_inner_25_0 = None
        model_9_m = None
        model_9_m_1 = None
        model_9_m_2 = None
        model_9_cv2_conv = self.model_9_cv2_conv(cat_4_f)
        cat_4_f = None
        model_9_cv2_bn = self.model_9_cv2_bn(model_9_cv2_conv)
        model_9_cv2_conv = None
        fake_dequant_inner_32_0 = self.fake_dequant_inner_32_0(model_9_cv2_bn)
        model_9_cv2_bn = None
        model_9_cv2_act = self.model_9_cv2_act(fake_dequant_inner_32_0)
        fake_dequant_inner_32_0 = None
        fake_quant_inner_24_0 = self.fake_quant_inner_24_0(model_9_cv2_act)
        model_9_cv2_act = None
        model_10_conv = self.model_10_conv(fake_quant_inner_24_0)
        fake_quant_inner_24_0 = None
        model_10_bn = self.model_10_bn(model_10_conv)
        model_10_conv = None
        fake_dequant_inner_33_0 = self.fake_dequant_inner_33_0(model_10_bn)
        model_10_bn = None
        model_10_act = self.model_10_act(fake_dequant_inner_33_0)
        fake_dequant_inner_33_0 = None
        fake_quant_inner_23_0 = self.fake_quant_inner_23_0(model_10_act)
        model_10_act = None
        model_11 = self.model_11(fake_quant_inner_23_0)
        cat_5_f = self.float_functional_simple_12.cat([model_11, fake_quant_inner_32_0], 1)
        model_11 = None
        fake_quant_inner_32_0 = None
        model_13_cv1_conv = self.model_13_cv1_conv(cat_5_f)
        model_13_cv1_bn = self.model_13_cv1_bn(model_13_cv1_conv)
        model_13_cv1_conv = None
        fake_dequant_inner_34_0 = self.fake_dequant_inner_34_0(model_13_cv1_bn)
        model_13_cv1_bn = None
        model_13_cv1_act = self.model_13_cv1_act(fake_dequant_inner_34_0)
        fake_dequant_inner_34_0 = None
        fake_quant_inner_22_0 = self.fake_quant_inner_22_0(model_13_cv1_act)
        model_13_cv1_act = None
        model_13_m_0_cv1_conv = self.model_13_m_0_cv1_conv(fake_quant_inner_22_0)
        fake_quant_inner_22_0 = None
        model_13_m_0_cv1_bn = self.model_13_m_0_cv1_bn(model_13_m_0_cv1_conv)
        model_13_m_0_cv1_conv = None
        fake_dequant_inner_35_0 = self.fake_dequant_inner_35_0(model_13_m_0_cv1_bn)
        model_13_m_0_cv1_bn = None
        model_13_m_0_cv1_act = self.model_13_m_0_cv1_act(fake_dequant_inner_35_0)
        fake_dequant_inner_35_0 = None
        fake_quant_inner_21_0 = self.fake_quant_inner_21_0(model_13_m_0_cv1_act)
        model_13_m_0_cv1_act = None
        model_13_m_0_cv2_conv = self.model_13_m_0_cv2_conv(fake_quant_inner_21_0)
        fake_quant_inner_21_0 = None
        model_13_m_0_cv2_bn = self.model_13_m_0_cv2_bn(model_13_m_0_cv2_conv)
        model_13_m_0_cv2_conv = None
        fake_dequant_inner_36_0 = self.fake_dequant_inner_36_0(model_13_m_0_cv2_bn)
        model_13_m_0_cv2_bn = None
        model_13_m_0_cv2_act = self.model_13_m_0_cv2_act(fake_dequant_inner_36_0)
        fake_dequant_inner_36_0 = None
        model_13_cv2_conv = self.model_13_cv2_conv(cat_5_f)
        cat_5_f = None
        model_13_cv2_bn = self.model_13_cv2_bn(model_13_cv2_conv)
        model_13_cv2_conv = None
        fake_dequant_inner_37_0 = self.fake_dequant_inner_37_0(model_13_cv2_bn)
        model_13_cv2_bn = None
        model_13_cv2_act = self.model_13_cv2_act(fake_dequant_inner_37_0)
        fake_dequant_inner_37_0 = None
        fake_quant_inner_19_0 = self.fake_quant_inner_19_0(model_13_cv2_act)
        model_13_cv2_act = None
        fake_quant_inner_20_0 = self.fake_quant_inner_20_0(model_13_m_0_cv2_act)
        model_13_m_0_cv2_act = None
        cat_6_f = self.float_functional_simple_13.cat([fake_quant_inner_20_0, fake_quant_inner_19_0], 1)
        fake_quant_inner_20_0 = None
        fake_quant_inner_19_0 = None
        model_13_cv3_conv = self.model_13_cv3_conv(cat_6_f)
        cat_6_f = None
        model_13_cv3_bn = self.model_13_cv3_bn(model_13_cv3_conv)
        model_13_cv3_conv = None
        fake_dequant_inner_38_0 = self.fake_dequant_inner_38_0(model_13_cv3_bn)
        model_13_cv3_bn = None
        model_13_cv3_act = self.model_13_cv3_act(fake_dequant_inner_38_0)
        fake_dequant_inner_38_0 = None
        fake_quant_inner_18_0 = self.fake_quant_inner_18_0(model_13_cv3_act)
        model_13_cv3_act = None
        model_14_conv = self.model_14_conv(fake_quant_inner_18_0)
        fake_quant_inner_18_0 = None
        model_14_bn = self.model_14_bn(model_14_conv)
        model_14_conv = None
        fake_dequant_inner_39_0 = self.fake_dequant_inner_39_0(model_14_bn)
        model_14_bn = None
        model_14_act = self.model_14_act(fake_dequant_inner_39_0)
        fake_dequant_inner_39_0 = None
        fake_quant_inner_17_0 = self.fake_quant_inner_17_0(model_14_act)
        model_14_act = None
        model_15 = self.model_15(fake_quant_inner_17_0)
        cat_7_f = self.float_functional_simple_14.cat([model_15, fake_quant_inner_42_0], 1)
        model_15 = None
        fake_quant_inner_42_0 = None
        model_17_cv1_conv = self.model_17_cv1_conv(cat_7_f)
        model_17_cv1_bn = self.model_17_cv1_bn(model_17_cv1_conv)
        model_17_cv1_conv = None
        fake_dequant_inner_40_0 = self.fake_dequant_inner_40_0(model_17_cv1_bn)
        model_17_cv1_bn = None
        model_17_cv1_act = self.model_17_cv1_act(fake_dequant_inner_40_0)
        fake_dequant_inner_40_0 = None
        fake_quant_inner_16_0 = self.fake_quant_inner_16_0(model_17_cv1_act)
        model_17_cv1_act = None
        model_17_m_0_cv1_conv = self.model_17_m_0_cv1_conv(fake_quant_inner_16_0)
        fake_quant_inner_16_0 = None
        model_17_m_0_cv1_bn = self.model_17_m_0_cv1_bn(model_17_m_0_cv1_conv)
        model_17_m_0_cv1_conv = None
        fake_dequant_inner_41_0 = self.fake_dequant_inner_41_0(model_17_m_0_cv1_bn)
        model_17_m_0_cv1_bn = None
        model_17_m_0_cv1_act = self.model_17_m_0_cv1_act(fake_dequant_inner_41_0)
        fake_dequant_inner_41_0 = None
        fake_quant_inner_15_0 = self.fake_quant_inner_15_0(model_17_m_0_cv1_act)
        model_17_m_0_cv1_act = None
        model_17_m_0_cv2_conv = self.model_17_m_0_cv2_conv(fake_quant_inner_15_0)
        fake_quant_inner_15_0 = None
        model_17_m_0_cv2_bn = self.model_17_m_0_cv2_bn(model_17_m_0_cv2_conv)
        model_17_m_0_cv2_conv = None
        fake_dequant_inner_42_0 = self.fake_dequant_inner_42_0(model_17_m_0_cv2_bn)
        model_17_m_0_cv2_bn = None
        model_17_m_0_cv2_act = self.model_17_m_0_cv2_act(fake_dequant_inner_42_0)
        fake_dequant_inner_42_0 = None
        model_17_cv2_conv = self.model_17_cv2_conv(cat_7_f)
        cat_7_f = None
        model_17_cv2_bn = self.model_17_cv2_bn(model_17_cv2_conv)
        model_17_cv2_conv = None
        fake_dequant_inner_43_0 = self.fake_dequant_inner_43_0(model_17_cv2_bn)
        model_17_cv2_bn = None
        model_17_cv2_act = self.model_17_cv2_act(fake_dequant_inner_43_0)
        fake_dequant_inner_43_0 = None
        fake_quant_inner_13_0 = self.fake_quant_inner_13_0(model_17_cv2_act)
        model_17_cv2_act = None
        fake_quant_inner_14_0 = self.fake_quant_inner_14_0(model_17_m_0_cv2_act)
        model_17_m_0_cv2_act = None
        cat_8_f = self.float_functional_simple_15.cat([fake_quant_inner_14_0, fake_quant_inner_13_0], 1)
        fake_quant_inner_14_0 = None
        fake_quant_inner_13_0 = None
        model_17_cv3_conv = self.model_17_cv3_conv(cat_8_f)
        cat_8_f = None
        model_17_cv3_bn = self.model_17_cv3_bn(model_17_cv3_conv)
        model_17_cv3_conv = None
        fake_dequant_inner_44_0 = self.fake_dequant_inner_44_0(model_17_cv3_bn)
        model_17_cv3_bn = None
        model_17_cv3_act = self.model_17_cv3_act(fake_dequant_inner_44_0)
        fake_dequant_inner_44_0 = None
        fake_quant_inner_12_0 = self.fake_quant_inner_12_0(model_17_cv3_act)
        model_17_cv3_act = None
        model_18_conv = self.model_18_conv(fake_quant_inner_12_0)
        model_18_bn = self.model_18_bn(model_18_conv)
        model_18_conv = None
        fake_dequant_inner_45_0 = self.fake_dequant_inner_45_0(model_18_bn)
        model_18_bn = None
        model_18_act = self.model_18_act(fake_dequant_inner_45_0)
        fake_dequant_inner_45_0 = None
        fake_quant_inner_11_0 = self.fake_quant_inner_11_0(model_18_act)
        model_18_act = None
        cat_9_f = self.float_functional_simple_16.cat([fake_quant_inner_11_0, fake_quant_inner_17_0], 1)
        fake_quant_inner_11_0 = None
        fake_quant_inner_17_0 = None
        model_20_cv1_conv = self.model_20_cv1_conv(cat_9_f)
        model_20_cv1_bn = self.model_20_cv1_bn(model_20_cv1_conv)
        model_20_cv1_conv = None
        fake_dequant_inner_46_0 = self.fake_dequant_inner_46_0(model_20_cv1_bn)
        model_20_cv1_bn = None
        model_20_cv1_act = self.model_20_cv1_act(fake_dequant_inner_46_0)
        fake_dequant_inner_46_0 = None
        fake_quant_inner_10_0 = self.fake_quant_inner_10_0(model_20_cv1_act)
        model_20_cv1_act = None
        model_20_m_0_cv1_conv = self.model_20_m_0_cv1_conv(fake_quant_inner_10_0)
        fake_quant_inner_10_0 = None
        model_20_m_0_cv1_bn = self.model_20_m_0_cv1_bn(model_20_m_0_cv1_conv)
        model_20_m_0_cv1_conv = None
        fake_dequant_inner_47_0 = self.fake_dequant_inner_47_0(model_20_m_0_cv1_bn)
        model_20_m_0_cv1_bn = None
        model_20_m_0_cv1_act = self.model_20_m_0_cv1_act(fake_dequant_inner_47_0)
        fake_dequant_inner_47_0 = None
        fake_quant_inner_9_0 = self.fake_quant_inner_9_0(model_20_m_0_cv1_act)
        model_20_m_0_cv1_act = None
        model_20_m_0_cv2_conv = self.model_20_m_0_cv2_conv(fake_quant_inner_9_0)
        fake_quant_inner_9_0 = None
        model_20_m_0_cv2_bn = self.model_20_m_0_cv2_bn(model_20_m_0_cv2_conv)
        model_20_m_0_cv2_conv = None
        fake_dequant_inner_48_0 = self.fake_dequant_inner_48_0(model_20_m_0_cv2_bn)
        model_20_m_0_cv2_bn = None
        model_20_m_0_cv2_act = self.model_20_m_0_cv2_act(fake_dequant_inner_48_0)
        fake_dequant_inner_48_0 = None
        model_20_cv2_conv = self.model_20_cv2_conv(cat_9_f)
        cat_9_f = None
        model_20_cv2_bn = self.model_20_cv2_bn(model_20_cv2_conv)
        model_20_cv2_conv = None
        fake_dequant_inner_49_0 = self.fake_dequant_inner_49_0(model_20_cv2_bn)
        model_20_cv2_bn = None
        model_20_cv2_act = self.model_20_cv2_act(fake_dequant_inner_49_0)
        fake_dequant_inner_49_0 = None
        fake_quant_inner_7_0 = self.fake_quant_inner_7_0(model_20_cv2_act)
        model_20_cv2_act = None
        fake_quant_inner_8_0 = self.fake_quant_inner_8_0(model_20_m_0_cv2_act)
        model_20_m_0_cv2_act = None
        cat_10_f = self.float_functional_simple_17.cat([fake_quant_inner_8_0, fake_quant_inner_7_0], 1)
        fake_quant_inner_8_0 = None
        fake_quant_inner_7_0 = None
        model_20_cv3_conv = self.model_20_cv3_conv(cat_10_f)
        cat_10_f = None
        model_20_cv3_bn = self.model_20_cv3_bn(model_20_cv3_conv)
        model_20_cv3_conv = None
        fake_dequant_inner_50_0 = self.fake_dequant_inner_50_0(model_20_cv3_bn)
        model_20_cv3_bn = None
        model_20_cv3_act = self.model_20_cv3_act(fake_dequant_inner_50_0)
        fake_dequant_inner_50_0 = None
        fake_quant_inner_6_0 = self.fake_quant_inner_6_0(model_20_cv3_act)
        model_20_cv3_act = None
        model_21_conv = self.model_21_conv(fake_quant_inner_6_0)
        model_21_bn = self.model_21_bn(model_21_conv)
        model_21_conv = None
        fake_dequant_inner_51_0 = self.fake_dequant_inner_51_0(model_21_bn)
        model_21_bn = None
        model_21_act = self.model_21_act(fake_dequant_inner_51_0)
        fake_dequant_inner_51_0 = None
        fake_quant_inner_5_0 = self.fake_quant_inner_5_0(model_21_act)
        model_21_act = None
        cat_11_f = self.float_functional_simple_18.cat([fake_quant_inner_5_0, fake_quant_inner_23_0], 1)
        fake_quant_inner_5_0 = None
        fake_quant_inner_23_0 = None
        model_23_cv1_conv = self.model_23_cv1_conv(cat_11_f)
        model_23_cv1_bn = self.model_23_cv1_bn(model_23_cv1_conv)
        model_23_cv1_conv = None
        fake_dequant_inner_52_0 = self.fake_dequant_inner_52_0(model_23_cv1_bn)
        model_23_cv1_bn = None
        model_23_cv1_act = self.model_23_cv1_act(fake_dequant_inner_52_0)
        fake_dequant_inner_52_0 = None
        fake_quant_inner_4_0 = self.fake_quant_inner_4_0(model_23_cv1_act)
        model_23_cv1_act = None
        model_23_m_0_cv1_conv = self.model_23_m_0_cv1_conv(fake_quant_inner_4_0)
        fake_quant_inner_4_0 = None
        model_23_m_0_cv1_bn = self.model_23_m_0_cv1_bn(model_23_m_0_cv1_conv)
        model_23_m_0_cv1_conv = None
        fake_dequant_inner_53_0 = self.fake_dequant_inner_53_0(model_23_m_0_cv1_bn)
        model_23_m_0_cv1_bn = None
        model_23_m_0_cv1_act = self.model_23_m_0_cv1_act(fake_dequant_inner_53_0)
        fake_dequant_inner_53_0 = None
        fake_quant_inner_3_0 = self.fake_quant_inner_3_0(model_23_m_0_cv1_act)
        model_23_m_0_cv1_act = None
        model_23_m_0_cv2_conv = self.model_23_m_0_cv2_conv(fake_quant_inner_3_0)
        fake_quant_inner_3_0 = None
        model_23_m_0_cv2_bn = self.model_23_m_0_cv2_bn(model_23_m_0_cv2_conv)
        model_23_m_0_cv2_conv = None
        fake_dequant_inner_54_0 = self.fake_dequant_inner_54_0(model_23_m_0_cv2_bn)
        model_23_m_0_cv2_bn = None
        model_23_m_0_cv2_act = self.model_23_m_0_cv2_act(fake_dequant_inner_54_0)
        fake_dequant_inner_54_0 = None
        model_23_cv2_conv = self.model_23_cv2_conv(cat_11_f)
        cat_11_f = None
        model_23_cv2_bn = self.model_23_cv2_bn(model_23_cv2_conv)
        model_23_cv2_conv = None
        fake_dequant_inner_55_0 = self.fake_dequant_inner_55_0(model_23_cv2_bn)
        model_23_cv2_bn = None
        model_23_cv2_act = self.model_23_cv2_act(fake_dequant_inner_55_0)
        fake_dequant_inner_55_0 = None
        fake_quant_inner_1_0 = self.fake_quant_inner_1_0(model_23_cv2_act)
        model_23_cv2_act = None
        fake_quant_inner_2_0 = self.fake_quant_inner_2_0(model_23_m_0_cv2_act)
        model_23_m_0_cv2_act = None
        cat_12_f = self.float_functional_simple_19.cat([fake_quant_inner_2_0, fake_quant_inner_1_0], 1)
        fake_quant_inner_2_0 = None
        fake_quant_inner_1_0 = None
        model_23_cv3_conv = self.model_23_cv3_conv(cat_12_f)
        cat_12_f = None
        model_23_cv3_bn = self.model_23_cv3_bn(model_23_cv3_conv)
        model_23_cv3_conv = None
        fake_dequant_inner_56_0 = self.fake_dequant_inner_56_0(model_23_cv3_bn)
        model_23_cv3_bn = None
        model_23_cv3_act = self.model_23_cv3_act(fake_dequant_inner_56_0)
        fake_dequant_inner_56_0 = None
        model_24_m_0 = self.model_24_m_0(fake_quant_inner_12_0)
        fake_quant_inner_12_0 = None
        shape_0_f = model_24_m_0.shape
        view_0_f = model_24_m_0.view(shape_0_f[0], 3, 85, shape_0_f[2], shape_0_f[3])
        model_24_m_0 = None
        shape_0_f = None
        permute_0_f = view_0_f.permute(0, 1, 3, 4, 2)
        view_0_f = None
        contiguous_0_f = permute_0_f.contiguous()
        permute_0_f = None
        model_24_m_1 = self.model_24_m_1(fake_quant_inner_6_0)
        fake_quant_inner_6_0 = None
        shape_1_f = model_24_m_1.shape
        view_1_f = model_24_m_1.view(shape_1_f[0], 3, 85, shape_1_f[2], shape_1_f[3])
        model_24_m_1 = None
        shape_1_f = None
        permute_1_f = view_1_f.permute(0, 1, 3, 4, 2)
        view_1_f = None
        contiguous_1_f = permute_1_f.contiguous()
        permute_1_f = None
        fake_quant_inner_0_0 = self.fake_quant_inner_0_0(model_23_cv3_act)
        model_23_cv3_act = None
        model_24_m_2 = self.model_24_m_2(fake_quant_inner_0_0)
        fake_quant_inner_0_0 = None
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

