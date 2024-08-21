build_params = {
    "datatype": "E4M3",
    "ic_dimension": 16,
    "oc_dimension": 16,
    "technology": "intel16",
    "clock_period": 1
}

sim_params = {
    "sims": "accelerator,customposit",
    "data_dir": "/sim/kzf/wbirch/data",
    "network": "resnet18",          # default network
    "layer": "layer2_0_downsample", # default layer
}

sweep_params = {
"datatypes": ['E4M3', 'BF16', 'P8_1'],
# "dimensions": [[16, 16], [16, 32], [32, 32]],
"dimensions": [16, 32],
"tests": {
        "resnet18": [
            "conv1",
            "layer1_0_conv1",
            "layer1_0_conv2",
            "layer1_1_conv1",
            "layer1_1_conv2",
            "layer2_0_downsample",
            "layer2_0_conv1",
            "layer2_0_conv2",
            "layer2_1_conv1",
            "layer2_1_conv2",
            "layer3_0_downsample",
            "layer3_0_conv1",
            "layer3_0_conv2",
            "layer3_1_conv1",
            "layer3_1_conv2",
            "layer4_0_downsample",
            "layer4_0_conv1",
            "layer4_0_conv2",
            "layer4_1_conv1",
            "layer4_1_conv2",
            "fc",
        ],
        "mobilebert": [
            "bottleneck_input_dense",
            "bottleneck_input_LayerNorm",
            # "bottleneck_attention_dense",
            # "bottleneck_attention_LayerNorm",
            "attention_self_query_layer",
            # "attention_self_key_layer",
            "attention_self_value_layer",
            # "attention_self_attention_scores_0",
            # "attention_self_attention_scores_1",
            "attention_self_attention_scores_2",
            # "attention_self_attention_scores_3",
            "attention_self_attention_probs_0",
            # "attention_self_attention_probs_1",
            # "attention_self_attention_probs_2",
            # "attention_self_attention_probs_3",
            "attention_self_context_layer_0",
            "attention_self_context_layer_1",
            "attention_self_context_layer_2",
            "attention_self_context_layer_3",
            "attention_output_dense",
            # "attention_output_LayerNorm",
            # "ffn_0_intermediate_dense",
            "ffn_0_output_dense",
            # "ffn_0_output_LayerNorm",
            "intermediate_dense",
            # "output_dense",
            # "output_LayerNorm",
            "output_bottleneck_dense",
            "output_bottleneck_LayerNorm",
            "classifier",
        ],
    }
}
