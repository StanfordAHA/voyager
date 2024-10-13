#!/bin/env xonsh

import os
import json
from deepdiff import DeepDiff
from pprint import pprint

def delete_nested_keys(data, key):
    # if the current element is a dict
    if isinstance(data, dict):
        # use list() to create a copy, avoiding modifying the dictionary while iterating
        for k in list(data.keys()):
            if k == key:
                del data[k]
            else:
                # recursively check the value of the current key
                delete_nested_keys(data[k], key)

    # if the current element is a list
    elif isinstance(data, list):
        for item in data:
            delete_nested_keys(item, key)

def collect_layers(networks: list, datatype: str):
    layers = {}
    for network in networks:
        layers[network] = {}
        layer_params_json = json.load(open(f"{os.path.dirname(__file__)}/../accel-src/test/compiler/networks/{network}/{datatype}/params.json"))
        layer_params = {}

        for lp in layer_params_json['params']:
            name = lp['name']
            delete_nested_keys(lp, "name")
            delete_nested_keys(lp, "memory")
            delete_nested_keys(lp, "node")
            exists = False
            # Deep compare this layer param with all other layers to see if it's a duplicate
            for layer_name, layer_val in layer_params.items():
                dif = DeepDiff(lp, layer_val, ignore_order=True)
                if not dif:
                    print(f"{network} duplicate layer removed: {name} == {layer_name}")
                    # increment multiplier
                    layers[network][layer_name] += 1
                    exists = True
                    break
            if not exists:
                layer_params.update({name: lp})
                # add layer name to the list of layers to run, value is the multiplier, initially 1
                layers[network][name] = 1

    return layers

if __name__ == "__main__":
    layers = collect_layers(["resnet18", "resnet50"], "E4M3")
    pprint(layers)

