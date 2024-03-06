from ptflops import get_model_complexity_info


def convert_number(num):
    suffixes = ['', 'K', 'M', 'B', 'T']
    suffix_index = 0
    while num >= 1000 and suffix_index < len(suffixes) - 1:
        suffix_index += 1  # Increment the suffix index
        num /= 1000.0  # Divide the number by 1000
    if suffix_index > 0:
        result = '{:.2f}{}'.format(num, suffixes[suffix_index])
    else:
        result = f"{num}"
    return result

def convert_to_percentage(num):
    percentage = num * 100
    return '{:.1f}%'.format(percentage)

def get_model_params(model, resolution):
    macs, params = get_model_complexity_info(
        model, 
        (3, resolution, resolution), 
        as_strings=False,
        print_per_layer_stat=False, 
        verbose=False
    )

    return (macs, params)