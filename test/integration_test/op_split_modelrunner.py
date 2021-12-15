# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import numpy as np
import tensorflow
import argparse

def test_inference(args):
    #
    # before
    #
    model = args.model
    with open(model, "rb") as fd:
        model_content = fd.read()

    interpreter = tensorflow.lite.Interpreter(
        model_content=model_content)
    interpreter.allocate_tensors()
    input_tensor_details = interpreter.get_input_details()[0]

    print("Creating random input...")
    input_tensor = np.array(100 * np.random.random_sample(
        input_tensor_details["shape"]),
                            dtype=input_tensor_details["dtype"])

    interpreter.set_tensor(input_tensor_details["index"], input_tensor)
    print("Invoking tf interpreter...")
    interpreter.invoke()

    num_of_outputs = len(interpreter.get_output_details())
    tflite_outputs = []
    for i in range(num_of_outputs):
        tflite_outputs.append(
            interpreter.get_tensor(
                interpreter.get_output_details()[i]["index"]))
    #
    # after
    #

    op_split_model = args.op_split_model
    with open(op_split_model, "rb") as fd:
        model_content = fd.read()

    interpreter = tensorflow.lite.Interpreter(
        model_content=model_content)
    interpreter.allocate_tensors()
    input_tensor_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_tensor_details["index"], input_tensor)
    print("Invoking tf interpreter...")
    interpreter.invoke()
    op_split_outputs = []
    for i in range(num_of_outputs):
        op_split_outputs.append(
            interpreter.get_tensor(
                interpreter.get_output_details()[i]["index"]))
    #
    # compare
    #
    print("Comparing outputs...")
    for i in range(num_of_outputs):
        print("Comparing output " + str(i) + "...")
        if np.any(tflite_outputs[i] != op_split_outputs[-i+1]):
            print("Number wrong: " +str(np.count_nonzero(tflite_outputs[i] != op_split_outputs[-i+1])) + " Out of " + str(tflite_outputs[i].size))
            print("Comparing output " + str(i) + "..." + str(tflite_outputs[i] - op_split_outputs[-i+1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="provide model tflite file")
    parser.add_argument("op_split_model", help="provide op-split model tflite file")
    args = parser.parse_args()

    test_inference(args)
