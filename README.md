# Generating weights and biases for MPSCNNConvolution

Converts Inception v3 batch-normalized weights into weights and biases for MPSCNNConvolution.

## Results

The generated weights and biases are close to the weights included in the MetalImageRecognition sample code. The max delta for the weights and biases are listed below.

```
conv                                     [max delta: w=0.000406 b=0.000037]
conv_1                                   [max delta: w=0.000016 b=0.000040]
conv_2                                   [max delta: w=0.000005 b=0.000022]
conv_3                                   [max delta: w=0.000010 b=0.000014]
conv_4                                   [max delta: w=0.000008 b=0.000013]
mixed_conv                               [max delta: w=0.000006 b=0.000009]
mixed_tower_conv                         [max delta: w=0.000007 b=0.000012]
mixed_tower_conv_1                       [max delta: w=0.000005 b=0.000007]
mixed_tower_1_conv                       [max delta: w=0.000013 b=0.000062]
mixed_tower_1_conv_1                     [max delta: w=0.000014 b=0.000048]
mixed_tower_1_conv_2                     [max delta: w=0.000018 b=0.000033]
mixed_tower_2_conv                       [max delta: w=0.000021 b=0.000042]
mixed_1_conv                             [max delta: w=0.000007 b=0.000015]
mixed_1_tower_conv                       [max delta: w=0.000010 b=0.000013]
mixed_1_tower_conv_1                     [max delta: w=0.000008 b=0.000008]
mixed_1_tower_1_conv                     [max delta: w=0.000017 b=0.000020]
mixed_1_tower_1_conv_1                   [max delta: w=0.000013 b=0.000030]
mixed_1_tower_1_conv_2                   [max delta: w=0.000011 b=0.000019]
mixed_1_tower_2_conv                     [max delta: w=0.000031 b=0.000074]
mixed_2_conv                             [max delta: w=0.000008 b=0.000012]
mixed_2_tower_conv                       [max delta: w=0.000013 b=0.000025]
mixed_2_tower_conv_1                     [max delta: w=0.000006 b=0.000009]
mixed_2_tower_1_conv                     [max delta: w=0.000012 b=0.000025]
mixed_2_tower_1_conv_1                   [max delta: w=0.000008 b=0.000015]
mixed_2_tower_1_conv_2                   [max delta: w=0.000008 b=0.000010]
mixed_2_tower_2_conv                     [max delta: w=0.000034 b=0.000042]
mixed_3_conv                             [max delta: w=0.000010 b=0.000018]
mixed_3_tower_conv                       [max delta: w=0.000013 b=0.000020]
mixed_3_tower_conv_1                     [max delta: w=0.000013 b=0.000022]
mixed_3_tower_conv_2                     [max delta: w=0.000022 b=0.000031]
mixed_4_conv                             [max delta: w=0.000010 b=0.000021]
mixed_4_tower_conv                       [max delta: w=0.000010 b=0.000014]
mixed_4_tower_conv_1                     [max delta: w=0.000011 b=0.000015]
mixed_4_tower_conv_2                     [max delta: w=0.000011 b=0.000015]
mixed_4_tower_1_conv                     [max delta: w=0.000019 b=0.000025]
mixed_4_tower_1_conv_1                   [max delta: w=0.000015 b=0.000021]
mixed_4_tower_1_conv_2                   [max delta: w=0.000016 b=0.000016]
mixed_4_tower_1_conv_3                   [max delta: w=0.000016 b=0.000021]
mixed_4_tower_1_conv_4                   [max delta: w=0.000014 b=0.000025]
mixed_4_tower_2_conv                     [max delta: w=0.000031 b=0.000089]
mixed_5_conv                             [max delta: w=0.000012 b=0.000013]
mixed_5_tower_conv                       [max delta: w=0.000017 b=0.000025]
mixed_5_tower_conv_1                     [max delta: w=0.000022 b=0.000020]
mixed_5_tower_conv_2                     [max delta: w=0.000013 b=0.000019]
mixed_5_tower_1_conv                     [max delta: w=0.000017 b=0.000022]
mixed_5_tower_1_conv_1                   [max delta: w=0.000019 b=0.000022]
mixed_5_tower_1_conv_2                   [max delta: w=0.000014 b=0.000019]
mixed_5_tower_1_conv_3                   [max delta: w=0.000014 b=0.000018]
mixed_5_tower_1_conv_4                   [max delta: w=0.000012 b=0.000017]
mixed_5_tower_2_conv                     [max delta: w=0.000032 b=0.000051]
mixed_6_conv                             [max delta: w=0.000011 b=0.000014]
mixed_6_tower_conv                       [max delta: w=0.000019 b=0.000025]
mixed_6_tower_conv_1                     [max delta: w=0.000031 b=0.000022]
mixed_6_tower_conv_2                     [max delta: w=0.000016 b=0.000023]
mixed_6_tower_1_conv                     [max delta: w=0.000014 b=0.000027]
mixed_6_tower_1_conv_1                   [max delta: w=0.000026 b=0.000037]
mixed_6_tower_1_conv_2                   [max delta: w=0.000018 b=0.000023]
mixed_6_tower_1_conv_3                   [max delta: w=0.000018 b=0.000031]
mixed_6_tower_1_conv_4                   [max delta: w=0.000018 b=0.000051]
mixed_6_tower_2_conv                     [max delta: w=0.000026 b=0.000034]
mixed_7_conv                             [max delta: w=0.000012 b=0.000017]
mixed_7_tower_conv                       [max delta: w=0.000017 b=0.000023]
mixed_7_tower_conv_1                     [max delta: w=0.000027 b=0.000034]
mixed_7_tower_conv_2                     [max delta: w=0.000020 b=0.000030]
mixed_7_tower_1_conv                     [max delta: w=0.000020 b=0.000027]
mixed_7_tower_1_conv_1                   [max delta: w=0.000027 b=0.000027]
mixed_7_tower_1_conv_2                   [max delta: w=0.000015 b=0.000026]
mixed_7_tower_1_conv_3                   [max delta: w=0.000015 b=0.000023]
mixed_7_tower_1_conv_4                   [max delta: w=0.000014 b=0.000030]
mixed_7_tower_2_conv                     [max delta: w=0.000025 b=0.000039]
mixed_8_tower_conv                       [max delta: w=0.000096 b=0.000074]
mixed_8_tower_conv_1                     [max delta: w=0.000046 b=0.000093]
mixed_8_tower_1_conv                     [max delta: w=0.000094 b=0.000088]
mixed_8_tower_1_conv_1                   [max delta: w=0.000020 b=0.000038]
mixed_8_tower_1_conv_2                   [max delta: w=0.000016 b=0.000034]
mixed_8_tower_1_conv_3                   [max delta: w=0.000026 b=0.000064]
mixed_9_conv                             [max delta: w=0.000023 b=0.000030]
mixed_9_tower_conv                       [max delta: w=0.000020 b=0.000028]
mixed_9_tower_mixed_conv                 [max delta: w=0.000038 b=0.000065]
mixed_9_tower_mixed_conv_1               [max delta: w=0.000044 b=0.000059]
mixed_9_tower_1_conv                     [max delta: w=0.000019 b=0.000031]
mixed_9_tower_1_conv_1                   [max delta: w=0.000016 b=0.000025]
mixed_9_tower_1_mixed_conv               [max delta: w=0.000025 b=0.000063]
mixed_9_tower_1_mixed_conv_1             [max delta: w=0.000026 b=0.000064]
mixed_9_tower_2_conv                     [max delta: w=0.000026 b=0.000043]
mixed_10_conv                            [max delta: w=0.000385 b=0.000087]
mixed_10_tower_conv                      [max delta: w=0.000059 b=0.000101]
mixed_10_tower_mixed_conv                [max delta: w=0.000436 b=0.000089]
mixed_10_tower_mixed_conv_1              [max delta: w=0.000492 b=0.000091]
mixed_10_tower_1_conv                    [max delta: w=0.000035 b=0.000059]
mixed_10_tower_1_conv_1                  [max delta: w=0.000029 b=0.000055]
mixed_10_tower_1_mixed_conv              [max delta: w=0.000100 b=0.000048]
mixed_10_tower_1_mixed_conv_1            [max delta: w=0.000155 b=0.000050]
mixed_10_tower_2_conv                    [max delta: w=0.000045 b=0.000027]
softmax                                  [max delta: w=0.000005 b=0.000005]
```

The predictions between the Tensorflow Batch-Normalized model, the original model and the generated model are close but not exact.
A comparison of 1000 ImageNet images is included in `prediction_comparison.txt`.
The top-1 and top-5 error rates for the original and generated models (vs the Tensorflow Batch-Normalized model) are:

```
Original  Top-1: 24.6%
Generated Top-1: 24.3%
Original  Top-5:  4.2%
Generated Top-5:  4.2%
```

## Dependencies

- [Python 2.7](https://www.python.org/)
- [numpy](http://www.numpy.org/)
- [Tensorflow](https://www.tensorflow.org/)

## Usage

Basic usage:

```
./convert.py
```

will download the Inception v3 weights into the `input` directory and generate weights and biases in the `output` directory. The file format follows the MetalImageRecognition sample code.

If you want to compare the generated values to the MetalImageRecognition values:

```
./convert.py --dat-dir=[path to MetalImageRecognition .dat files]
```

Other options:

- `--help`: Display the help message.
- `--inception3-url`: Inception v3 model URL.
- `--input-dir`: Directory to download the Inception v3 model.
- `--output-dir`: Directory to generate weights and biases.
- `--dat-dir`: Directory of MetalImageRecognition .dat files.

## Implementation Details

Apple's MetalImageRecognition README provides this note for converting batch-normalized weights into weights and biases:

```
The weights for this particular network were batch normalized but for inference we may use :

w = 𝛄 / √(s + 0.001), b = ß - ( A * m )

s: variance
m: mean
𝛄: gamma
ß: beta

w: weights of a feature channel
b: bias of a feature channel 

for every feature channel separately to get the corresponding weights and biases
```

In the paper 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift' by Sergey Ioffe and Christian Szegedy (https://arxiv.org/pdf/1502.03167v3.pdf), we can use Algorithm 2, Output, Step&nbsp;11 to derive:

![Weight = \frac{\gamma}{\sqrt{Var[x]+0.001}} * Weight_{BN}](http://mathurl.com/z7snq3z.png)


![Bias = \beta - (\frac{\gamma}{\sqrt{Var[x]+0.001}})*E[x]](http://mathurl.com/zo4shhf.png)

