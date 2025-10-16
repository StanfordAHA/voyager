#include "test/common/Tiling.h"

#include "spdlog/spdlog.h"
#include "test/common/VerificationTypes.h"

std::ostream& operator<<(std::ostream& os, const Tiling& tiling) {
  os << "Loops: " << std::endl;
  for (int i = 0; i < 2; i++) {
    os << "  " << i << ": ";
    for (int j = 0; j < 6; j++) {
      os << tiling.loops[i][j] << " ";
    }
    os << std::endl;
  }
  os << "X Loop Index: " << tiling.x_loop_index[0] << " "
     << tiling.x_loop_index[1] << std::endl;
  os << "Y Loop Index: " << tiling.y_loop_index[0] << " "
     << tiling.y_loop_index[1] << std::endl;
  os << "Reduction Loop Index: " << tiling.reduction_loop_index[0] << " "
     << tiling.reduction_loop_index[1] << std::endl;
  os << "Weight Loop Index: " << tiling.weight_loop_index[0] << " "
     << tiling.weight_loop_index[1] << std::endl;
  os << "FX Index: " << tiling.fx_index << std::endl;
  os << "FY Index: " << tiling.fy_index << std::endl;
  os << "Weight Reuse Index: " << tiling.weight_reuse_index[0] << " "
     << tiling.weight_reuse_index[1] << std::endl;
  os << "Stride: " << tiling.stride << std::endl;
  os << "Replication: " << tiling.replication << std::endl;
  return os;
}

Tiling get_tiling(const Operation& operation, bool hack_tiling) {
  const auto param = operation.param;
  const auto op_list = get_op_list(param);
  const auto first_op = op_list[0];

  std::string operation_name = first_op.name();

  // get environment variable
  const char* env_var = std::getenv("MANUAL_TILING");
  bool manual_tiling = env_var ? std::stoi(env_var) : false;

  const char* zircon_hardcoded_tiling_env = std::getenv("ZIRCON_HARDCODED_TILING");
  bool zircon_hardcoded_tiling = zircon_hardcoded_tiling_env && std::stoi(zircon_hardcoded_tiling_env) == 1;

  const char* zircon_fx_fy_stride_workaround_env = std::getenv("ZIRCON_FX_FY_STRIDE_WORKAROUND");
  bool zircon_fx_fy_stride_workaround = zircon_fx_fy_stride_workaround_env && std::stoi(zircon_fx_fy_stride_workaround_env) == 1;

  const char* zircon_cgra_psum_workaround_env = std::getenv("ZIRCON_CGRA_PSUM_WORKAROUND");
  bool zircon_cgra_psum_workaround = zircon_cgra_psum_workaround_env && std::stoi(zircon_cgra_psum_workaround_env) == 1;

  const char* k_dim_host_tiling_env = std::getenv("K_DIM_HOST_TILING");
  bool k_dim_host_tiling = k_dim_host_tiling_env && std::stoi(k_dim_host_tiling_env) == 1;

  const char* zircon_input_act_padding_workaround_env = std::getenv("ZIRCON_INPUT_ACT_PADDING_WORKAROUND");
  bool zircon_input_act_padding_workaround = zircon_input_act_padding_workaround_env && std::stoi(zircon_input_act_padding_workaround_env) == 1;

  const char* zircon_inner_loop_reduction_workaround_env = std::getenv("ZIRCON_INNER_LOOP_REDUCTION_WORKAROUND");
  bool zircon_inner_loop_reduction_workaround = zircon_inner_loop_reduction_workaround_env && std::stoi(zircon_inner_loop_reduction_workaround_env) == 1;

  const char* zircon_gemm_x_dim_host_tiling_env = std::getenv("ZIRCON_GEMM_X_DIM_HOST_TILING");
  bool zircon_gemm_x_dim_host_tiling = zircon_gemm_x_dim_host_tiling_env && std::stoi(zircon_gemm_x_dim_host_tiling_env) == 1;

  const char* zircon_gemm_x_dimm_host_tiling_slice_length_env = std::getenv("X_DIM_HOST_TILING_SLICE_LENGTH");
  int zircon_gemm_x_dim_host_tiling_slice_length = 0;
  if (zircon_gemm_x_dimm_host_tiling_slice_length_env) {
    zircon_gemm_x_dim_host_tiling_slice_length = std::stoi(zircon_gemm_x_dimm_host_tiling_slice_length_env);
  }

  Tiling tiling;
  if (manual_tiling || !operation.has_valid_tiling) {
    if (first_op.target() == "conv2d" || first_op.target() == "conv2d_mx") {
      tiling = get_conv2d_tiling(first_op);
    } else {
      tiling = get_linear_tiling(first_op);
    }
  } else if(zircon_hardcoded_tiling && hack_tiling) {
      tiling = get_zircon_hardcoded_tiling(first_op);
  } else if (zircon_fx_fy_stride_workaround && hack_tiling) {
        tiling = get_zircon_fx_fy_stride_workaround_tiling(first_op);
  } else {
    tiling = get_interstellar_tiling(operation.tiling);
    if (first_op.kwargs().contains("stride")) {
      auto stride = first_op.kwargs().at("stride").int_list().values();
      tiling.stride = stride[0];
    } else {
      tiling.stride = 1;
    }

    // Inner loop reduction workaround for Zircon.
    // The workaround ensures there is a non-trivial reduction loop in the inner level. It moves the outer channel loop to inner level.
    // This step is needed to avoid a bug in the Zircon MU.
    if (zircon_inner_loop_reduction_workaround && hack_tiling) {
      tiling.loops[1][tiling.reduction_loop_index[1]] *= tiling.loops[0][tiling.reduction_loop_index[0]];
      tiling.loops[0][tiling.reduction_loop_index[0]] = 1;
    }

    // PSUM workaround for Zircon. MU reduction loop bounds must be 1 for MU to work.
    if (zircon_cgra_psum_workaround && hack_tiling) {
      tiling.loops[0][tiling.reduction_loop_index[0]] = 1;
      tiling.loops[1][tiling.reduction_loop_index[1]] = 1;
    }

    if (k_dim_host_tiling && hack_tiling) {
        const char* num_k_host_tiling_kernels_env = std::getenv("NUM_K_HOST_TILING_KERNELS");

        printf("NUM_K_HOST_TILING_KERNELS: %s\n", num_k_host_tiling_kernels_env);

        int num_k_host_tiling_kernels;

        if (num_k_host_tiling_kernels_env) {
          num_k_host_tiling_kernels = std::stoi(num_k_host_tiling_kernels_env);
        } else {
          throw std::runtime_error(
              "Zircon K HOST TILING workaround requires NUM_K_HOST_TILING_KERNELS "
              "environment variable to be set.");
        }

      // NOTE: Assumption is that entire K loop is in k2 and it is divisible by num_k_host_tiling_kernels, which is the case for conv5 layers in Resnet18 on Zircon
      if ((tiling.loops[0][tiling.weight_loop_index[0]] % num_k_host_tiling_kernels != 0) || (tiling.loops[1][tiling.weight_loop_index[1]] != 1)) {
        throw std::runtime_error(
            "Zircon K HOST TILING workaround by default assumes weight loop to be divisible by NUM_K_HOST_TILING_KERNELS and the second weight loop to be 1."
            "Code modifications will be necessary to proceed.");
      }

      tiling.loops[0][tiling.weight_loop_index[0]] = tiling.loops[0][tiling.weight_loop_index[0]] / num_k_host_tiling_kernels;
    }

    // Workaround to pad inputs to account for Zircon MU's inability to handle some layer shapes
    if (zircon_input_act_padding_workaround && hack_tiling){
      const char* zircon_input_act_padding_workaround_size_env = std::getenv("ZIRCON_INPUT_ACT_PADDING_WORKAROUND_SIZE");
      int zircon_input_act_padding_workaround_size;

      if (zircon_input_act_padding_workaround_size_env) {
        zircon_input_act_padding_workaround_size = std::stoi(zircon_input_act_padding_workaround_size_env);
      } else {
        throw std::runtime_error(
            "Zircon input padding workaround requires ZIRCON_INPUT_ACT_PADDING_WORKAROUND_SIZE "
            "environment variable to be set.");
      }

      // NOTE: Assumption is that the entire input image is in the inner loop, which is the case for conv5 layers in Resnet18 on Zircon
      if ((tiling.loops[0][tiling.x_loop_index[0]] != 1) || (tiling.loops[0][tiling.y_loop_index[0]] != 1)) {
        throw std::runtime_error(
            "Zircon input padding workaround by default assumes outer x and y loops to be 1. "
            "Code modifications will be necessary to proceed.");
      }

      tiling.loops[1][tiling.x_loop_index[1]] += zircon_input_act_padding_workaround_size;
      tiling.loops[1][tiling.y_loop_index[1]] += zircon_input_act_padding_workaround_size;

    }

    // This is a HACK for x = 3136, used for Resnet18 conv1 in Zircon
    if (zircon_gemm_x_dim_host_tiling && hack_tiling && (zircon_gemm_x_dim_host_tiling_slice_length == 3136)) {
      tiling.loops[0][tiling.x_loop_index[0]] = 14;
      tiling.loops[1][tiling.x_loop_index[1]] = 224;
    }
  }

  return tiling;
}


Tiling get_zircon_hardcoded_tiling(const codegen::OpOverload param) {
  Tiling tiling;

  printf("Using ZIRCON HARDCODED TILING for %s\n", param.name().c_str());
  const auto kwargs = param.kwargs();
  const auto input = kwargs.at("input").tensor();
  const auto weight = kwargs.contains("weight") ? kwargs.at("weight").tensor() : kwargs.at("other").tensor();
  int stride = 1;

  if (kwargs.contains("stride")) {
    const auto strides = kwargs.at("stride").int_list().values();
    stride = strides[0];
  }

  const auto input_shape = get_shape(input);
  const auto weight_shape = get_shape(weight);

  // submodule (conv1 im2col GEMM), tiled into 4 sub-kernels along x dim
  if (input_shape[2] == 192 && input_shape[1] == 12544 && input_shape[0] == 1 &&
      weight_shape[1] == 64 && weight_shape[0] == 192 && stride == 1) {

    tiling = {
          .loops = {{1, 1, 32, 1, 1, 1}, {3, 1, 1, 1, 2, 98}},
          .x_loop_index = {2, 5},
          .y_loop_index = {0, 1},
          .reduction_loop_index = {3, 0},
          .weight_loop_index = {1, 4},
          .fx_index = 3,
          .fy_index = 2,
          .weight_reuse_index = {5, 5},
          .stride = 1,
          .replication = false,
      };

  // submodule_3 (conv2_x)
 } else if (input_shape[3] == 64 && input_shape[1] == 56 && input_shape[2] == 56 &&
      weight_shape[3] == 64 && weight_shape[0] == 3 && weight_shape[1] == 3 && stride == 1) {

    tiling = {
          .loops = {{2, 4, 2, 1, 1, 1}, {1, 1, 3, 3, 14, 28}},
          .x_loop_index = {2, 5},
          .y_loop_index = {1, 4},
          .reduction_loop_index = {3, 0},
          .weight_loop_index = {0, 1},
          .fx_index = 3,
          .fy_index = 2,
          .weight_reuse_index = {4, 5},
          .stride = 1,
          .replication = false,
      };

  // conv2d_mx_default_6 (conv3_x)
  } else if (input_shape[3] == 128 && input_shape[1] == 28 && input_shape[2] == 28 &&
      weight_shape[3] == 128 && weight_shape[0] == 3 && weight_shape[1] == 3 && stride == 1) {

        tiling = {
          .loops = {{4, 2, 2, 2, 1, 1}, {1, 1, 3, 3, 14, 14}},
          .x_loop_index = {2, 5},
          .y_loop_index = {1, 4},
          .reduction_loop_index = {3, 0},
          .weight_loop_index = {0, 1},
          .fx_index = 3,
          .fy_index = 2,
          .weight_reuse_index = {4, 5},
          .stride = 1,
          .replication = false,
      };

  // submodule_7 (conv3_downsample)
  } else if (input_shape[3] == 64 && input_shape[1] == 56 && input_shape[2] == 56 &&
      weight_shape[3] == 128 && weight_shape[0] == 1 && weight_shape[1] == 1 && stride == 2) {

      tiling = {
          .loops = {{2, 2, 1, 1, 1, 1}, {1, 4, 1, 1, 14, 14}},
          .x_loop_index = {1, 5},
          .y_loop_index = {0, 4},
          .reduction_loop_index = {3, 0},
          .weight_loop_index = {2, 1},
          .fx_index = 3,
          .fy_index = 2,
          .weight_reuse_index = {4, 5},
          .stride = 2,
          .replication = false,
      };


  // conv2d_mx_default_11 (conv4_x)
  } else if (input_shape[3] == 256 && input_shape[1] == 14 && input_shape[2] == 14 &&
      weight_shape[3] == 256 && weight_shape[0] == 3 && weight_shape[1] == 3 && stride == 1) {

        tiling = {
          .loops = {{1, 1, 8, 4, 1, 1}, {1, 1, 3, 3, 14, 14}},
          .x_loop_index = {1, 5},
          .y_loop_index = {0, 4},
          .reduction_loop_index = {3, 0},
          .weight_loop_index = {2, 1},
          .fx_index = 3,
          .fy_index = 2,
          .weight_reuse_index = {4, 5},
          .stride = 1,
          .replication = false,
      };

  } else if (input_shape[3] == 256 && input_shape[1] == 14 && input_shape[2] == 14 &&
      weight_shape[3] == 512 && weight_shape[0] == 1 && weight_shape[1] == 1 && stride == 2) {

        tiling = {
          .loops = {{1, 1, 2, 2, 1, 1}, {2, 1, 1, 8, 8, 8}},
          .x_loop_index = {1, 5},
          .y_loop_index = {0, 4},
          .reduction_loop_index = {3, 0},
          .weight_loop_index = {2, 3},
          .fx_index = 2,
          .fy_index = 1,
          .weight_reuse_index = {4, 5},
          .stride = 2,
          .replication = false,
      };
  } else {
     throw std::runtime_error("Zircon hardcoded tiling not implemented for this layer!");
  }

  return tiling;
}


Tiling get_zircon_fx_fy_stride_workaround_tiling(const codegen::OpOverload param) {
  Tiling tiling;

  printf("Using kernel and stride hack for %s\n", param.name().c_str());
  const auto kwargs = param.kwargs();
  const auto input = kwargs.at("input").tensor();
  const auto weight = kwargs.at("weight").tensor();
  const auto strides = kwargs.at("stride").int_list().values();

  const auto input_shape = get_shape(input);
  const auto weight_shape = get_shape(weight);
  int stride = strides[0];

  // conv4 downsample
  if (input_shape[3] == 128 && input_shape[1] == 28 && input_shape[2] == 28 &&
      weight_shape[3] == 256 && weight_shape[0] == 1 && weight_shape[1] == 1 && stride == 2) {

    tiling = {
          .loops = {{1, 1, 8, 2, 1, 1}, {1, 1, 3, 3, 28, 28}},
          .x_loop_index = {1, 5},
          .y_loop_index = {0, 4},
          .reduction_loop_index = {3, 0},
          .weight_loop_index = {2, 1},
          .fx_index = 3,
          .fy_index = 2,
          .weight_reuse_index = {4, 5},
          .stride = 1,
          .replication = false,
      };
  // conv5 downsample
  } else if (input_shape[3] == 256 && input_shape[1] == 14 && input_shape[2] == 14 &&
      weight_shape[3] == 512 && weight_shape[0] == 1 && weight_shape[1] == 1 && stride == 2) {

        tiling = {
          .loops = {{1, 1, 16, 4, 1, 1}, {1, 1, 3, 3, 14, 14}},
          .x_loop_index = {1, 5},
          .y_loop_index = {0, 4},
          .reduction_loop_index = {3, 0},
          .weight_loop_index = {2, 1},
          .fx_index = 3,
          .fy_index = 2,
          .weight_reuse_index = {4, 5},
          .stride = 1,
          .replication = false,
      };
  } else {
     throw std::runtime_error("Zircon fx, fy, stride workaround not implemented for this layer!");
  }

  return tiling;
}

Tiling get_interstellar_tiling(const voyager::Tiling& tiling) {
  Tiling accelerator_tiling;

  // Interstellar does not emit tilings with replication
  accelerator_tiling.replication = false;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      accelerator_tiling.loops[i][j] = 1;
    }
  }

  accelerator_tiling.fx_index = -1;
  accelerator_tiling.fy_index = -1;
  for (int i = 0; i < 2; i++) {
    accelerator_tiling.x_loop_index[i] = -1;
    accelerator_tiling.y_loop_index[i] = -1;
    accelerator_tiling.reduction_loop_index[i] = -1;
    accelerator_tiling.weight_loop_index[i] = -1;
  }

  int loop_index = 5;
  // L1 level
  for (int i = 0; i < tiling.level_tilings(0).loop_bounds_size(); i++) {
    if (tiling.level_tilings(0).loop_bounds(i).loop() == voyager::Loop::IC) {
      // set this later
      accelerator_tiling.loops[1][0] =
          tiling.level_tilings(0).loop_bounds(i).bound();
      accelerator_tiling.reduction_loop_index[1] = 0;
    } else {
      // all other loops need to be set in reverse order
      accelerator_tiling.loops[1][loop_index] =
          tiling.level_tilings(0).loop_bounds(i).bound();
      if (tiling.level_tilings(0).loop_bounds(i).loop() == voyager::Loop::FX) {
        accelerator_tiling.fx_index = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::Loop::FY) {
        accelerator_tiling.fy_index = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::Loop::OC) {
        accelerator_tiling.weight_loop_index[1] = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::Loop::OX) {
        accelerator_tiling.x_loop_index[1] = loop_index;
      } else if (tiling.level_tilings(0).loop_bounds(i).loop() ==
                 voyager::Loop::OY) {
        accelerator_tiling.y_loop_index[1] = loop_index;
      }

      loop_index--;
    }
  }

  // set any unset loop indices
  while (loop_index >= 1) {
    if (accelerator_tiling.fx_index == -1) {
      accelerator_tiling.fx_index = loop_index;
    } else if (accelerator_tiling.fy_index == -1) {
      accelerator_tiling.fy_index = loop_index;
    } else if (accelerator_tiling.weight_loop_index[1] == -1) {
      accelerator_tiling.weight_loop_index[1] = loop_index;
    } else if (accelerator_tiling.x_loop_index[1] == -1) {
      accelerator_tiling.x_loop_index[1] = loop_index;
    } else if (accelerator_tiling.y_loop_index[1] == -1) {
      accelerator_tiling.y_loop_index[1] = loop_index;
    }
    loop_index--;
  }

  // if reduction loop is not set, set it to 0
  if (accelerator_tiling.reduction_loop_index[1] == -1) {
    accelerator_tiling.reduction_loop_index[1] = 0;
  }

  for (int i = loop_index; i < 6; i++) {
    if (accelerator_tiling.fx_index == -1) {
      accelerator_tiling.fx_index = 5 - i;
    } else if (accelerator_tiling.fy_index == -1) {
      accelerator_tiling.fy_index = 5 - i;
    } else if (accelerator_tiling.weight_loop_index[1] == -1) {
      accelerator_tiling.weight_loop_index[1] = 5 - i;
    } else if (accelerator_tiling.x_loop_index[1] == -1) {
      accelerator_tiling.x_loop_index[1] = 5 - i;
    } else if (accelerator_tiling.y_loop_index[1] == -1) {
      accelerator_tiling.y_loop_index[1] = 5 - i;
    } else if (accelerator_tiling.reduction_loop_index[1] == -1) {
      accelerator_tiling.reduction_loop_index[1] = 5 - i;
    }
  }

  // set weight reuse loop index depending on if x or y are the innermost loops
  if (accelerator_tiling.x_loop_index[1] == 5 ||
      accelerator_tiling.y_loop_index[1] == 5) {
    accelerator_tiling.weight_reuse_index[1] = 5;
    accelerator_tiling.weight_reuse_index[0] = 5;
  }
  if (accelerator_tiling.x_loop_index[1] == 4 ||
      accelerator_tiling.y_loop_index[1] == 4) {
    accelerator_tiling.weight_reuse_index[0] = 4;
  }

  // L2 level
  int offset = 0;
  if (tiling.level_tilings(1).loop_bounds_size() > 0 &&
      tiling.level_tilings(1).loop_bounds(0).loop() != voyager::Loop::IC) {
    // if the first loop is not IC, then we need to manually set the IC loop to
    // 1
    accelerator_tiling.loops[0][3] = 1;
    accelerator_tiling.reduction_loop_index[0] = 3;
    offset = 1;
  }

  for (int i = 0; i < tiling.level_tilings(1).loop_bounds_size(); i++) {
    accelerator_tiling.loops[0][3 - offset - i] =
        tiling.level_tilings(1).loop_bounds(i).bound();
    if (tiling.level_tilings(1).loop_bounds(i).loop() == voyager::Loop::OC) {
      accelerator_tiling.weight_loop_index[0] = 3 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::Loop::OX) {
      accelerator_tiling.x_loop_index[0] = 3 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::Loop::OY) {
      accelerator_tiling.y_loop_index[0] = 3 - offset - i;
    } else if (tiling.level_tilings(1).loop_bounds(i).loop() ==
               voyager::Loop::IC) {
      accelerator_tiling.reduction_loop_index[0] = 3 - offset - i;
    }
  }

  // set any unset loop indices
  for (int i = tiling.level_tilings(1).loop_bounds_size() - 1; i < 3 - offset;
       i++) {
    accelerator_tiling.loops[0][2 - i - offset] = 1;
    if (accelerator_tiling.weight_loop_index[0] == -1) {
      accelerator_tiling.weight_loop_index[0] = 2 - i - offset;
    } else if (accelerator_tiling.x_loop_index[0] == -1) {
      accelerator_tiling.x_loop_index[0] = 2 - i - offset;
    } else if (accelerator_tiling.y_loop_index[0] == -1) {
      accelerator_tiling.y_loop_index[0] = 2 - i - offset;
    } else if (accelerator_tiling.reduction_loop_index[0] == -1) {
      accelerator_tiling.reduction_loop_index[0] = 2 - i - offset;
    }
  }

  return accelerator_tiling;
}

Tiling get_conv2d_tiling(const codegen::OpOverload param) {
  const auto kwargs = param.kwargs();

  const auto input = kwargs.at("input").tensor();
  const auto weight = kwargs.at("weight").tensor();
  const auto padding = kwargs.at("padding").int_list().values();
  const auto dilation = kwargs.at("dilation").int_list().values();
  const auto strides = kwargs.at("stride").int_list().values();

  const auto input_shape = get_shape(input);
  const auto weight_shape = get_shape(weight);

  const int output_height = (input_shape[1] + 2 * padding[0] -
                             dilation[0] * (weight_shape[0] - 1) - 1) /
                                strides[0] +
                            1;
  const int output_width = (input_shape[2] + 2 * padding[1] -
                            dilation[1] * (weight_shape[1] - 1) - 1) /
                               strides[1] +
                           1;


  std::vector<int> output_shape = {
      output_height,
      output_width,
      input_shape[3],
      weight_shape[3],
  };


  const int oc_unroll = OC_DIMENSION;
  const int ic_unroll = IC_DIMENSION;

  int x1 = 1, y1 = 1, k1 = 1;
  int x0 = output_shape[2];
  int y0 = output_shape[1];
  int k0 = weight_shape[3] / oc_unroll;
  int c0 = weight_shape[2] / ic_unroll;
  int fx = weight_shape[1];
  int fy = weight_shape[0];
  int stride = strides[0];

  // conv1
  if (input_shape[3] == 3 && input_shape[1] == 224 && input_shape[2] == 224 &&
      weight_shape[3] == 64 && weight_shape[0] == 7 && weight_shape[1] == 7) {
    int fx;
    if (IC_DIMENSION == 4) {
      fx = 7;
    } else if (IC_DIMENSION == 8) {
      fx = 4;
    } else if (IC_DIMENSION == 16) {
      fx = 2;
    } else if (IC_DIMENSION == 32) {
      fx = 1;
    } else {
      throw std::runtime_error("replication not supported for IC_DIMENSION=" +
                               std::to_string(IC_DIMENSION));
    }

    Tiling tiling = {
        .loops = {{7, 7, 2, 1, 1, 1}, {1, 2, 7, fx, 16, 16}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = stride,
        .replication = true,
    };

    if (IC_DIMENSION < 16) {
      tiling.loops[1][5] /= 2;
      tiling.loops[0][0] *= 2;
    }

    if (OC_DIMENSION < 16) {
      tiling.loops[0][tiling.weight_loop_index[0]] *= (16 / OC_DIMENSION);
    } else if (OC_DIMENSION > 16) {
      int div_factor = OC_DIMENSION / 16;
      while (tiling.loops[0][tiling.weight_loop_index[0]] > 1 &&
             div_factor > 1) {
        tiling.loops[0][tiling.weight_loop_index[0]] /= 2;
        div_factor /= 2;
      }
      while (tiling.loops[1][tiling.weight_loop_index[1]] > 1 &&
             div_factor > 1) {
        tiling.loops[1][tiling.weight_loop_index[1]] /= 2;
        div_factor /= 2;
      }

      if (div_factor > 1) {
        spdlog::error("OC_DIMENSION is not a multiple of 16\n");
        exit(1);
      }
    }

    return tiling;
  }

  // Reduce OC0 to meet weight buffer constraint
  while (fx * fy * k0 * ic_unroll > WEIGHT_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else {
      spdlog::error("Weight buffer is too small\n");
      exit(1);
    }
  }

  // Reduce OC0 to meet weight buffer constraint
  while (fx * fy * k0 * ic_unroll > WEIGHT_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else {
      spdlog::error("Weight buffer is too small\n");
      exit(1);
    }
  }

  // Reduce X0 and Y0 to meet input buffer constraint. We are not counting
  // stride here because of the hardware implementation
  while (true) {
    int ix = x0 * stride + fx - 1;
    int iy = y0 * stride + fy - 1;
    if (ix * iy <= INPUT_BUFFER_SIZE) {
      break;
    }
    if (x0 % 2 == 0 && y0 % 2 == 0) {
      x0 /= 2;
      x1 *= 2;
      y0 /= 2;
      y1 *= 2;
    } else {
      spdlog::error("Input buffer is too small\n");
      exit(1);
    }
  }

  // Reduce either OC0, or OX0 and OY0, to meet accumulation buffer
  // constraint
  const int max_k0 = k0;
  while (x0 * y0 * k0 > ACCUM_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else if (x0 % 2 == 0 && y0 % 2 == 0) {
      x0 /= 2;
      x1 *= 2;
      y0 /= 2;
      y1 *= 2;
      // Since we are reducing both x0 and y0, there is a chance we can
      // increase k0
      if (k0 * 2 <= max_k0) {
        k0 *= 2;
        k1 /= 2;
      }
    } else {
      spdlog::error("Accumulation buffer is too small\n");
      exit(1);
    }
  }

  return {
      .loops = {{x1, y1, k1, c0, 1, 1}, {1, k0, fy, fx, y0, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = stride,
  };
}

Tiling get_linear_tiling(const codegen::OpOverload op) {
  const auto kwargs = op.kwargs();
  const auto input_shape = get_shape(kwargs.at("input").tensor());

  bool is_matmul = op.target().find("matmul") != std::string::npos;
  std::string weight_key = is_matmul ? "other" : "weight";
  const auto weight_shape = get_shape(kwargs.at(weight_key).tensor());

  // Generate tiling using unroll factor of 16
  const int oc_unroll = OC_DIMENSION;
  const int ic_unroll = IC_DIMENSION;

  int x1 = 1, k1 = 1, c1 = 1;
  int x0 = get_size(input_shape) / input_shape.back();
  int k0 = weight_shape[0] / oc_unroll;
  int c0 = weight_shape[1] / ic_unroll;

  // torch.matmul weight is also an activation, thus does not need to be
  // transposed
  if (op.target() == "matmul" || op.target() == "matmul_mx") {
    int size = weight_shape.size();
    c0 = weight_shape[size - 2] / ic_unroll;
    k0 = weight_shape[size - 1] / oc_unroll;
  }

  // Loop indices cannot exceed 1024 (10-bit)
  while (x0 >= 1024 || x0 * c0 > INPUT_BUFFER_SIZE) {
    if (x0 % 2 == 0) {
      x0 /= 2;
      x1 *= 2;
    } else if (c0 % 2 == 0) {
      c0 /= 2;
      c1 *= 2;
    } else {
      spdlog::error("Input buffer is too small\n");
      exit(1);
    }
  }

  while (k0 * c0 * ic_unroll > WEIGHT_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else if (c0 % 2 == 0) {
      c0 /= 2;
      c1 *= 2;
    } else {
      spdlog::error("Weight buffer is too small\n");
      exit(1);
    }
  }

  while (x0 * k0 > ACCUM_BUFFER_SIZE) {
    if (k0 % 2 == 0) {
      k0 /= 2;
      k1 *= 2;
    } else if (x0 % 2 == 0) {
      x0 /= 2;
      x1 *= 2;
    } else {
      spdlog::error("Accumulation buffer is too small\n");
      exit(1);
    }
  }

  return {
      .loops = {{x1, 1, k1, c1, 1, 1}, {c0, k0, 1, 1, 1, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = 1,
      .replication = false,
  };
}

Tiling get_pool2d_tiling(const codegen::OpOverload op) {
  const auto kwargs = op.kwargs();
  const auto input_shape = get_shape(kwargs.at("input").tensor());

  int Y = input_shape[1];
  int X = input_shape[2];
  int K = input_shape[3];

  int x0, y0, stride, padding, x1, y1, k0, actual_padding;

  if (kwargs.contains("output_size")) {
    const auto output_size = kwargs.at("output_size").int_list().values();
    int output_h = output_size[0];
    int output_w = output_size[1];

    stride = X / output_h;
    x0 = X - (output_h - 1) * stride;
    y0 = Y - (output_w - 1) * stride;

    x1 = X / x0;
    y1 = Y / y0;
    k0 = K / OC_DIMENSION;
    actual_padding = 0;
  } else {
    const auto kernel_size = kwargs.at("kernel_size").int_list().values();
    const auto strides = kwargs.at("stride").int_list().values();

    y0 = kernel_size[0];
    x0 = kernel_size[1];
    stride = strides[0];

    // calculate ouptut dimension (ignoring padding, which will be handled in
    // hw)
    x1 = (X + 2 * padding - x0) / stride + 1;
    y1 = (Y + 2 * padding - y0) / stride + 1;
    // pytorch assumes padding on all direction, not all of the values are used
    actual_padding = (x1 - 1) * stride + x0 - X;
    k0 = K / OC_DIMENSION;

  }
  return {
      .loops = {{x1, y1, 1, 1, 1, 1}, {1, k0, 1, 1, y0, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = stride,
      .replication = false,
  };
}
