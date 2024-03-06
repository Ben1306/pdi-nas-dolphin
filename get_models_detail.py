from efficientnet import EfficientNet
import os 

output_path = "outputs/models_detail"

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 288, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

possible_k = [1, 2, 3, 4, 5, 6]

S_strings = [
    "[1, 8, 1, 1, 3]",
    "[k, 16, 2, 2, 3]",
    "[k, 24, 2, 2, 5]",
    "[k, 64, 3, 2, 3]",
    "[k, 88, 3, 1, 5]",
    "[k, 128, 4, 2, 5]",
    "[k, 200, 1, 1, 3]"
]

M_strings = [
    "[1, 16, 1, 1, 3]",
    "[k, 24, 2, 2, 3]",
    "[k, 32, 2, 2, 5]",
    "[k, 72, 3, 2, 3]",
    "[k, 96, 3, 1, 5]",
    "[k, 160, 4, 2, 5]",
    "[k, 280, 1, 1, 3]"
]

L_strings = [
    "[1, 16, 1, 1, 3]",
    "[k, 24, 2, 2, 3]",
    "[k, 40, 2, 2, 5]",
    "[k, 80, 3, 2, 3]",
    "[k, 112, 3, 1, 5]",
    "[k, 192, 4, 2, 5]",
    "[k, 320, 1, 1, 3]"
]

def get_params(k, size="L"):
    if size == "S":
        return [
            # k, out_channels(c), repeats(t), stride(s), kernel_size(k)
            [1, 8, 1, 1, 3],
            [k, 16, 2, 2, 3],
            [k, 24, 2, 2, 5],
            [k, 64, 3, 2, 3],
            [k, 88, 3, 1, 5],
            [k, 128, 4, 2, 5],
            [k, 200, 1, 1, 3],
        ]
    elif size == "M":
        return [
            # k, out_channels(c), repeats(t), stride(s), kernel_size(k)
            [1, 16, 1, 1, 3],
            [k, 24, 2, 2, 3],
            [k, 32, 2, 2, 5],
            [k, 72, 3, 2, 3],
            [k, 96, 3, 1, 5],
            [k, 160, 4, 2, 5],
            [k, 280, 1, 1, 3],
        ]
    elif size == "L":
        return [
            # k, out_channels(c), repeats(t), stride(s), kernel_size(k)
            [1, 16, 1, 1, 3],
            [k, 24, 2, 2, 3],
            [k, 40, 2, 2, 5],
            [k, 80, 3, 2, 3],
            [k, 112, 3, 1, 5],
            [k, 192, 4, 2, 5],
            [k, 320, 1, 1, 3],
        ]
    else:
        return False
    
def get_model_params(model):

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mbblocks_detail = []
    for (input,output,params) in zip(model.mb_block_input_sizes, model.mb_block_output_sizes, model.mb_block_parameters):
        mbblocks_detail.append((input,output,params))

    return (model_params, mbblocks_detail)


phi = scale_values["b0"][0]
resolution = scale_values["b0"][1]
dropout = scale_values["b0"][2]
output_class = 10 # for imagenette
# ImageNet has 1000 classes

def write_model_variations_info():
    models_info = []

    for k in possible_k:

        infos = []
        model_S = EfficientNet(phi, resolution, dropout, get_params(k, "S"), alpha, beta, output_class)
        model_M = EfficientNet(phi, resolution, dropout, get_params(k, "M"), alpha, beta, output_class)
        model_L = EfficientNet(phi, resolution, dropout, get_params(k, "L"), alpha, beta, output_class)

        infos.append(k)
        infos.append(get_model_params(model_S))
        infos.append(get_model_params(model_M))
        infos.append(get_model_params(model_L))

        models_info.append(infos)
    
    
    # Create intermediate directories if they do not exist
    os.makedirs(output_path, exist_ok=True)
    # Generate a document with the information collected
    with open(os.path.join(output_path, "models_summary.txt"), "w") as f:
        f.write("########################################################\n")
        f.write("############# EfficientNet Models Summary ############\n")
        f.write("######################################################## \n\n")

        f.write("We test different values of the expand ration k = {1,2,4,6}\n")
        f.write("For each value of k, we compare 3 model size (depending on the number of output channels of each MBBlock layer)\n\n")

        f.write("Model size MBBlocks layer descriptions: S, M, L\n\n")
        f.write("expand ratio, out_channels, repeats, stride, kernel_size\n\n")

        f.write("| {:^20} | {:^20} | {:^20} |\n".format("Model S", "Model M", "Model L"))
        f.write("| {:^20} | {:^20} | {:^20} |\n".format("","",""))
        for (S_s,M_s,L_s) in zip(S_strings, M_strings, L_strings):
            f.write("| {:^20} | {:^20} | {:^20} |\n".format(S_s, M_s, L_s))


        f.write("\n\n\n")

        f.write("########################################################\n")
        f.write("######################## Results #######################\n")
        f.write("######################################################## \n\n")

        for infos in models_info:
            f.write(f"Expand ration k = {infos[0]}\n")
            f.write(f"Input = (channels, width, height) | Output = (channels, width, height)\n\n")

            f.write("| {:^20} | {:^80} | {:^80} | {:^80} |\n".format("", "Model Config S", "Model Config M", "Model Config L"))
            f.write("| {:^20} | {:^80} | {:^80} | {:^80} |\n".format("","","",""))


            f.write("| {:^20} | {:^80} | {:^80} | {:^80} |\n".format("Total parameters:", infos[1][0], infos[2][0], infos[3][0]))
            f.write("| {:^20} | {:^80} | {:^80} | {:^80} |\n".format("","","",""))
            for i, (S,M,L) in enumerate(zip(infos[1][1], infos[2][1], infos[3][1])):
                f.write("| {:^20} | {:^80} | {:^80} | {:^80} |\n".format(
                    f"MBBlock {i}",
                    "{:^20} | {:^25} | {:^25}".format(f"Params: {S[2]}", f"Input: {S[0]}", f"Output: {S[1]}"),
                    "{:^20} | {:^25} | {:^25}".format(f"Params: {M[2]}", f"Input: {M[0]}", f"Output: {M[1]}"),
                    "{:^20} | {:^25} | {:^25}".format(f"Params: {L[2]}", f"Input: {L[0]}", f"Output: {L[1]}")
                ))
                
            f.write("\n\n\n")


if __name__ == "__main__":
    write_model_variations_info()