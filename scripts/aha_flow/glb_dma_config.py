import os
# This code is used to generate the GLB affine controller config for a specific tiling pattern in a neural network layer.

vanilla_arg_indices_dict = {
        "X0": 0,
        "Y0": 1,
        "X1": 2,
        "Y1": 3,
        "K1": 4,
        "K2": 5
    }


mha_arg_indices_dict = {
        "D1": 0,
        "D2": 1,
        "N0": 2,
        "N1": 3,
        "H0": 4,
        "H1": 5
    }

mha_concat_arg_indices_dict = {
        "D": 0,
        "N": 1,
        "H": 2
    }

def get_address(x0, y0, x1, y1, k1, k2, X0, Y0, X1, Y1, K1, K2, K0=32, broadcast_dims: list = []):
    # Handle broadcasting by setting the corresponding loop variable to 0
    if 'Y' in broadcast_dims or 'y' in broadcast_dims:
        y = 0
    else:
        y = y1 * Y0 + y0

    if 'X' in broadcast_dims or 'x' in broadcast_dims:
        x = 0
    else:
        x = x1 * X0 + x0

    if 'K' in broadcast_dims or 'k' in broadcast_dims:
        k = 0
    else:
        k = k2 * K1 * K0 + k1 * K0

    addr = y * (X1 * X0 * K2 * K1 * K0) + x * (K2 * K1 * K0) + k
    return addr


def get_address_mha_permute(d1, d2, n0, n1, h0, h1, D1, D2, N0, N1, H0, H1, D0=32, broadcast_dims: list = []):
    if 'D' in broadcast_dims or 'd' in broadcast_dims:
        d = 0
    else:
        d = d2 * D1 * D0 + d1 * D0

    if 'N' in broadcast_dims or 'n' in broadcast_dims:
        n = 0
    else:
        n = n1 * N0 + n0

    if 'H' in broadcast_dims or 'h' in broadcast_dims:
        h = 0
    else:
        h = h1 * H0 + h0

    head_size = D2 * D1 * D0
    seq_len = N0 * N1
    addr = h * (seq_len * head_size) + n * head_size + d

    return addr

def get_address_mha_concat(h, n, d, H, N, D, D0=32):
    addr = n * H * D*D0 + h * D*D0 + d*D0
    return addr

def get_address_wrapper(addr_args: list, loop_bounds: list, broadcast_dims: list = [], mha_permute: bool = False, num_attn_heads: int = 12, mha_concat: bool = False):
    """
    Returns the address based on the provided arguments.
    The order of the arguments should be [x0, y0, x1, y1, k1, k2].

    For MHA permute, the order should be [d1, d2, n0, n1, h0, h1].

    For MHA concat, the order should be h, n, d.
    """

    if not mha_concat:
        if len(addr_args) != 6:
            raise ValueError("addr_args must contain exactly 6 elements: [x0, y0, x1, y1, k1, k2]. If doing multi-head attention permute, the order should be [d1, d2, n0, n1, h0, h1].")
        if len(loop_bounds) != 6:
            raise ValueError("loop_bounds must contain exactly 6 elements: [X0, Y0, X1, Y1, K1, K2]. If doing multi-head attention permute, the order should be [D1, D2, N0, N1, H0, H1].")
    else:
        if len(addr_args) != 3:
            raise ValueError("addr_args must contain exactly 3 elements: [h, n, d] for MHA concat.")
        if len(loop_bounds) != 3:
            raise ValueError("loop_bounds must contain exactly 3 elements: [H, N, D] for MHA concat.")

    if mha_concat:
        d, n, h = addr_args
        D, N, H = loop_bounds
        return get_address_mha_concat(h, n, d, H, N, D)

    if mha_permute:
        d1, d2, n0, n1, h0, h1 = addr_args
        D1, D2, N0, N1, H0, H1 = loop_bounds
        return get_address_mha_permute(d1, d2, n0, n1, h0, h1, D1, D2, N0, N1, H0, H1, broadcast_dims=broadcast_dims)

    x0, y0, x1, y1, k1, k2 = addr_args
    X0, Y0, X1, Y1, K1, K2 = loop_bounds
    return get_address(x0, y0, x1, y1, k1, k2, X0, Y0, X1, Y1, K1, K2, broadcast_dims=broadcast_dims)

def print_addr_map(X0, Y0, X1, Y1, K1, K2, mha_permute: bool = False, num_attn_heads: int = 12):
    for k2 in range(0, K2):
        for k1 in range(0, K1):
            for y1 in range(0, Y1):
                for x1 in range(0, X1):
                    for y0 in range(0, Y0):
                        for x0 in range(0, X0):
                            addr = get_address(x0, y0, x1, y1, k1, k2, X0, Y0, X1, Y1, K1, K2)
                            print(f"addr: {addr}, bank index: {int(addr/32)}, y1: {y1}, x1: {x1}, k2: {k2}, k1: {k1}, y0: {y0}, x0: {x0}")


def get_dimensionality(loop_bounds):
    """
    Returns the dimensionality of the address space based on the loop bounds.
    """
    return sum(1 for dim in loop_bounds if dim > 1)


def trim_dimensionality(strides, loop_bounds, loop_order, arg_indices_dict, mha_permute: bool = False):
    """
    Trims the strides to only include dimensions that are greater than 1.
    """
    trimmed_strides = []
    trimmed_extents = []

    for idx, loop in enumerate(loop_order):
        if loop_bounds[arg_indices_dict[loop]] > 1:
            trimmed_strides.append(strides[idx])
            trimmed_extents.append(loop_bounds[arg_indices_dict[loop]])
    return trimmed_strides, trimmed_extents


def compute_strides(loop_order, loop_bounds, arg_indices_dict, broadcast_dims: list = [], mha_permute: bool = False, num_attn_heads: int = 12, mha_concat: bool = False):
    """
    Computes the data addresss strides for the given loop order and loop bounds.
    Does so by calculating the derivative of the address function with respect to each loop variable. E.g.,
    # dX0 = (K1 * K0)/32
    # dY0 = (get_address(0, 0, 0, 0, 1, 0) - get_address(0, 0, 0, 0, 0, X0 - 1))/32
    # dX1 = (get_address(0, 0, 1, 0, 0, 0) - get_address(0, 0, 0, 0, Y0 - 1, X0 - 1))/32
    # dY1 = (get_address(0, 1, 0, 0, 0, 0) - get_address(0, 0, X1 - 1, 0, Y0 - 1, X0 - 1))/32
    # dK1 = (get_address(0, 0, 0, 1, 0, 0) - get_address(0, Y1 - 1, X1 - 1, 0, Y0 - 1, X0 - 1))/32

    # The addresss function is defined as: addr = y * (X1 * X0 * K2 * K1 * K0) + x * (K2 * K1 * K0) + k
    """

    k_dim_host_tiling = "K_DIM_HOST_TILING" in os.environ and os.environ["K_DIM_HOST_TILING"] == "1"
    if k_dim_host_tiling:
        assert "NUM_K_HOST_TILING_KERNELS" in os.environ, "NUM_K_HOST_TILING_KERNELS environment variable must be set for K_DIM_HOST_TILING"
        num_k_host_tiling_kernels = int(os.environ.get("NUM_K_HOST_TILING_KERNELS"))
        # UNDO the change made in the tiling file. i.e. in Tiling.cc.
        # This is because the tiling file is generated with the assumption that K2 is divided by the number of host tiling kernels, but we need to restore the original K2 value for the stride calculation
        # to ensure data is stored in correct GLB addresses across all host tiling kernels.
        loop_bounds[5] = loop_bounds[5] * num_k_host_tiling_kernels


    strides = [0] * len(loop_order)
    for idx, loop in enumerate(loop_order):
        addr_args_1 = [0] * len(arg_indices_dict)
        addr_args_1[arg_indices_dict[loop]] = 1

        addr_args_0 = [0] * len(arg_indices_dict)
        # Now loop over everything beneath and set it to its max value in args_0
        for inner_idx in range(idx):
            addr_args_0[arg_indices_dict[loop_order[inner_idx]]] = loop_bounds[arg_indices_dict[loop_order[inner_idx]]] - 1

        addr_1 = get_address_wrapper(addr_args_1, loop_bounds, broadcast_dims=broadcast_dims, mha_permute=mha_permute, num_attn_heads=num_attn_heads, mha_concat=mha_concat)
        addr_0 = get_address_wrapper(addr_args_0, loop_bounds, broadcast_dims=broadcast_dims, mha_permute=mha_permute, num_attn_heads=num_attn_heads, mha_concat=mha_concat)

        # Divide by 32 to account for each "word" being 32 bytes in our MU address space. Dividing by 32 yields the index into the bank
        # This implies that a stride of 1 here means that the next address is 32 bytes away in the MU address space, as intended.
        # This effect is acheieved because in map.c, the data stride is first multiplied by 2 (to account for CGRA word being 2 bytes), then in E64 mode, all strides are multiplied by 4. So overall
        # the stride is +8 in the GLB bank address space, which translates to +32 in the MU address space.
        strides[idx] = (addr_1 - addr_0) // 32

    if k_dim_host_tiling:
        loop_bounds[5] = loop_bounds[5] // num_k_host_tiling_kernels # Restore K2 value

    return strides


def map_d_loops(dr, k_r_list, orig_loop_order_containment):
    k1_r = k_r_list[0]
    k2_r = k_r_list[1]

    # Initialize
    d1 = 1
    d2 = 1

    # K1_r can completely contain dr
    if dr <= k1_r:
        k1_r //= dr
        d1 = dr
        dr = 1
        orig_loop_order_containment['K1'].append('D1')
    # Need to split dr into d1 and d2
    else:
        d1 = k1_r
        k1_r = 1
        dr //= d1
        k2_r //= dr
        d2 = dr
        orig_loop_order_containment['K1'].append('D1')
        orig_loop_order_containment['K2'].append('D2')

    k_r_list[0] = k1_r
    k_r_list[1] = k2_r

    return [d1, d2]

def map_h_loops(hr, k_r_list, orig_loop_order_containment):
    k1_r = k_r_list[0]
    k2_r = k_r_list[1]

    # Initialize
    h0 = 1
    h1 = 1

    # K1_r can completely contain hr
    if hr <= k1_r:
        k1_r //= hr
        h0 = hr
        hr = 1
        orig_loop_order_containment['K1'].append('H0')
    # Need to split hr into h1 and h0
    else:
        h0 = k1_r
        k1_r = 1
        hr //= h0
        k2_r //= hr
        h1 = hr
        orig_loop_order_containment['K1'].append('H0')
        orig_loop_order_containment['K2'].append('H1')

    k_r_list[0] = k1_r
    k_r_list[1] = k2_r

    return [h0, h1]


def compress_mha_loop_bounds(mha_loop_bounds, mha_loop_order):
    loop_order_dict = {}
    for idx, loop in enumerate(mha_loop_order):
        loop_order_dict[loop] = idx

    if (loop_order_dict['D2'] - loop_order_dict['D1']) == 1:
        # Can abosrb D2 into D1
        mha_loop_bounds[mha_arg_indices_dict['D1']] *= mha_loop_bounds[mha_arg_indices_dict['D2']]
        mha_loop_bounds[mha_arg_indices_dict['D2']] = 1

    if (loop_order_dict['H1'] - loop_order_dict['H0']) == 1:
        # Can abosrb H1 into H0
        mha_loop_bounds[mha_arg_indices_dict['H0']] *= mha_loop_bounds[mha_arg_indices_dict['H1']]
        mha_loop_bounds[mha_arg_indices_dict['H1']] = 1

    if (loop_order_dict['N1'] - loop_order_dict['N0']) == 1:
        # Can abosrb N1 into N0
        mha_loop_bounds[mha_arg_indices_dict['N0']] *= mha_loop_bounds[mha_arg_indices_dict['N1']]
        mha_loop_bounds[mha_arg_indices_dict['N1']] = 1

def map_mha_loops(orig_loop_bounds: list, orig_loop_order, K0: int = 32, num_attn_heads: int = 12):
    X0, Y0, X1, Y1, K1, K2 = orig_loop_bounds

    orig_loop_order_containment = {}
    for loop in orig_loop_order:
        orig_loop_order_containment[loop] = []

    # Need to return mha_loop_bounds and mha_loop_order
    mha_loop_bounds = [0] * 6 # D1, D2, N0, N1, H0, H1
    mha_loop_order = []

    # N (seq_len) loops map directly to X loops
    N1 = X1
    N0 = X0
    orig_loop_order_containment['X1'].append('N1')
    orig_loop_order_containment['X0'].append('N0')

    head_size = (K2 * K1 * K0) // num_attn_heads

    dr = head_size // K0  # D loops to map
    hr = num_attn_heads  # H loops to map

    k_r_list = [K1, K2]
    d_loops = map_d_loops(dr, k_r_list, orig_loop_order_containment)
    h_loops = map_h_loops(hr, k_r_list, orig_loop_order_containment)

    # Populate the loop bounds
    mha_loop_bounds[0] = d_loops[0]  # D1
    mha_loop_bounds[1] = d_loops[1]  # D2
    mha_loop_bounds[2] = N0          # N0
    mha_loop_bounds[3] = N1          # N1
    mha_loop_bounds[4] = h_loops[0]  # H0
    mha_loop_bounds[5] = h_loops[1]  # H1


    # Establish the loop order
    for loop in orig_loop_order:
        contained_loops = orig_loop_order_containment[loop]
        for contained_loop in contained_loops:
            mha_loop_order.append(contained_loop)

    # Add any missing loops (they are 1 and won't appear in the containment mapping)
    for loop in mha_arg_indices_dict.keys():
        if loop not in mha_loop_order:
            mha_loop_order.append(loop)

    # After establishing the order, need to compress loop bounds wherever possible (i.e. collapse multiple levels into a single level)
    compress_mha_loop_bounds(mha_loop_bounds, mha_loop_order)

    return mha_loop_bounds, mha_loop_order



def get_glb_dma_config_helper(loop_order, loop_bounds, broadcast_dims: list = [], mha_permute: bool = False, num_attn_heads: int = 12, mha_concat: bool = False):

    arg_indices_dict = vanilla_arg_indices_dict
    if mha_permute:
        arg_indices_dict = mha_arg_indices_dict
    elif mha_concat:
        arg_indices_dict = mha_concat_arg_indices_dict

    strides = compute_strides(loop_order, loop_bounds, arg_indices_dict, broadcast_dims=broadcast_dims, mha_permute=mha_permute, num_attn_heads=num_attn_heads, mha_concat=mha_concat)
    trimmed_strides, trimmed_extents = trim_dimensionality(strides, loop_bounds, loop_order, arg_indices_dict, mha_permute)
    dimensionality = get_dimensionality(loop_bounds)
    assert len(trimmed_strides) == dimensionality
    assert len(trimmed_extents) == dimensionality

    return dimensionality, trimmed_strides, trimmed_extents


def get_glb_dma_config(output_tiling_filepath: str, zircon_fx_fy_stride_workaround: bool = False, zircon_input_act_padding_workaround: bool = False, mha_permute: bool = False, num_attn_heads: int = 1, broadcast_dims: list = []):
    """
    Reads the tiling file and returns the GLB DMA config.
    The tiling file should contain the loop order and loop bounds in a specific format.
    """

    loop_order = []
    loop_bounds = [0] * 6  # X0, Y0, X1, Y1, K1, K2

    with open(output_tiling_filepath, 'r') as f:
        lines = f.readlines()
        outer_loops = list(map(int, lines[1].strip().split('0: ')[1].split(' ')))
        inner_loops = list(map(int, lines[2].strip().split('1: ')[1].split(' ')))

        x_loop_indices = list(map(int, lines[3].strip().split('X Loop Index: ')[1].split(' ')))
        y_loop_indices = list(map(int, lines[4].strip().split('Y Loop Index: ')[1].split(' ')))
        k_loop_indices = list(map(int, lines[6].strip().split('Weight Loop Index: ')[1].split(' ')))

        loop_bounds[0] = inner_loops[x_loop_indices[1]]  # X0
        loop_bounds[1] = inner_loops[y_loop_indices[1]]  # Y0
        loop_bounds[2] = outer_loops[x_loop_indices[0]]  # X1
        loop_bounds[3] = outer_loops[y_loop_indices[0]]  # Y1
        loop_bounds[4] = inner_loops[k_loop_indices[1]]  # K1
        loop_bounds[5] = outer_loops[k_loop_indices[0]]  # K2

        if zircon_fx_fy_stride_workaround:
            loop_bounds[0] = loop_bounds[0] // 2 # divide X0 by 2 to account for the hack; actual output is 2x smaller than what MU produces
            loop_bounds[1] = loop_bounds[1] // 2 # divide Y0 by 2 to account for the hack; actual output is 2x smaller than what MU produces

        if zircon_input_act_padding_workaround:
            assert "ZIRCON_INPUT_ACT_PADDING_WORKAROUND_SIZE" in os.environ, "ZIRCON_INPUT_ACT_PADDING_WORKAROUND_SIZE environment variable must be set for ZIRCON_INPUT_ACT_PADDING_WORKAROUND"
            zircon_input_act_padding_workaround_size = int(os.environ.get("ZIRCON_INPUT_ACT_PADDING_WORKAROUND_SIZE", 0))
            # This file needs to ignore the padding to produce the correct addresses to store REAL data in GLB (i.e. the padded output gets filtered)
            loop_bounds[0] = loop_bounds[0] - zircon_input_act_padding_workaround_size
            loop_bounds[1] = loop_bounds[1] - zircon_input_act_padding_workaround_size

        # Construct loop order based on the indices
        # Map variable names to their values
        outer_vars_dict = {"X1": x_loop_indices[0], "Y1": y_loop_indices[0], "K2": k_loop_indices[0]}
        inner_vars_dict = {"X0": x_loop_indices[1], "Y0": y_loop_indices[1], "K1": k_loop_indices[1]}

        # Sort by values (descending), extract keys
        outer_loop_order = [k for k, v in sorted(outer_vars_dict.items(), key=lambda item: item[1], reverse=True)]
        inner_loop_order = [k for k, v in sorted(inner_vars_dict.items(), key=lambda item: item[1], reverse=True)]

        # Combine the loop orders
        loop_order = inner_loop_order + outer_loop_order

    if mha_permute:
        mha_loop_bounds, mha_loop_order = map_mha_loops(loop_bounds, loop_order, num_attn_heads=num_attn_heads)
        return get_glb_dma_config_helper(mha_loop_order, mha_loop_bounds, broadcast_dims=broadcast_dims, mha_permute=mha_permute, num_attn_heads=num_attn_heads)

    return get_glb_dma_config_helper(loop_order, loop_bounds, broadcast_dims=broadcast_dims)

if __name__ == "__main__":
    dimensionality, strides, extents = get_glb_dma_config("/aha/voyager/compiled_collateral/bert-matmul_mx_12/output_tiling.txt")
    print(f"Dimensionality: {dimensionality}")
    print(f"Strides: {strides}")
    print(f"Extents: {extents}")