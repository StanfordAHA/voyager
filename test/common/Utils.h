#pragma once
#define NO_SYSC

#include <spdlog/spdlog.h>

#include <any>
#include <fstream>
#include <iostream>
#include <bitset>
#include <cstring>  // for std::memcpy

#include "test/common/Tiling.h"

int validateMapping(Tiling tiling);

template <typename TA, typename TB>
float compare_arrays(std::any matrixA, std::string matrixA_name,
                     std::any matrixB, std::string matrixB_name, size_t size,
                     std::string filename_in, std::string suffix, bool doublePrecision, int size_of_typeB) {

  std::string filename = filename_in + suffix;
  spdlog::info("Writing comparison between {} and {} to file: {}\n",
               matrixA_name, matrixB_name, filename);
  std::ofstream diffFile(filename);
  diffFile << matrixA_name << " vs. " << matrixB_name << std::endl;


  // std::ofstream gold_data_file;
  // // Delete the file if it exists
  // std::string gold_data_filename = "gold" + suffix;
  // std::remove(gold_data_filename.c_str());
  // gold_data_file.open(gold_data_filename, std::ios::app);
  // if (!gold_data_file.is_open()) {
  //   spdlog::error("Failed to open {} for writing.", gold_data_filename);
  //   return -1;
  // }
  // spdlog::info("Writing gold data to file: {}\n", gold_data_filename);


  // std::string gold_binary_filename = "gold_data" + suffix + "raw";
  // // Delete the file if it exists
  // std::remove(gold_binary_filename.c_str());
  // std::ofstream gold_binary_file(gold_binary_filename, std::ios::binary);
  // spdlog::info("Writing gold binary data to file: {}\n", gold_binary_filename);

  // Records absolute differences
  int abs_diff_buckets[5] = {0, 0, 0, 0, 0};
  // Records relative differences
  int rel_diff_buckets[5] = {0, 0, 0, 0, 0};

  double always_zero = 0.0;

  TA *matrixA_ptr = std::any_cast<TA *>(matrixA);
  TB *matrixB_ptr = std::any_cast<TB *>(matrixB);

  for (int index = 0; index < size; index++) {
    // Calculate absolute difference
    float a = matrixA_ptr[index];
    float b = matrixB_ptr[index];
    always_zero += abs(a) + abs(b);
    float abs_diff = abs(a - b);

    // Write gold data out to text file
    // const unsigned char* ptr = reinterpret_cast<const unsigned char*>(&matrixB_ptr[index]);
    // for (size_t i = 0; i < size_of_typeB; ++i) {
    //     gold_data_file << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ptr[size_of_typeB - 1 - i]);
    // }
    // gold_data_file << std::endl;

    // write gold data out to binary (raw) file
    // const uint8_t* ptr2 = reinterpret_cast<const uint8_t*>(matrixB_ptr) + index * size_of_typeB;
    // // Loop over data in 2-byte chunks
    // for (size_t i = 0; i + 1 < size_of_typeB; i += 2) {
    //     uint16_t value;
    //     std::memcpy(&value, ptr2 + i, sizeof(uint16_t));  // safely extract 2 bytes

    //     uint8_t high_byte = (value >> 8) & 0xFF;
    //     uint8_t low_byte  = value & 0xFF;

    //     gold_binary_file.put(static_cast<char>(high_byte));
    //     gold_binary_file.put(static_cast<char>(low_byte));
    // }

    // Write the two values + error scale indicator to file
    diffFile << a << " vs. " << b << " ";
    for (float i = 0.001; i < abs_diff; i *= 10.0) {
      diffFile << "*";
    }
    diffFile << std::endl;

    if (abs_diff < 0.001) {
      abs_diff_buckets[0]++;
    }
    if (abs_diff < 0.01) {
      abs_diff_buckets[1]++;
    }
    if (abs_diff < 0.1) {
      abs_diff_buckets[2]++;
    }
    if (abs_diff < 1) {
      abs_diff_buckets[3]++;
    } else if (!std::isinf(abs_diff) && !std::isnan(abs_diff)) {
      abs_diff_buckets[4]++;
    }

    // Does not fully protect against overflow, but lets not over engineer
    if ((a == 0 && b == 0) || std::isinf(a) || std::isinf(b)) {
      rel_diff_buckets[0]++;
      rel_diff_buckets[1]++;
      rel_diff_buckets[2]++;
      rel_diff_buckets[3]++;
      continue;
    } else {
      // See https://en.wikipedia.org/wiki/Relative_change_and_difference
      float rel_diff = abs_diff / ((abs(a) + abs(b)) / 2);
      if (rel_diff < 0.001) {
        rel_diff_buckets[0]++;
      }
      if (rel_diff < 0.01) {
        rel_diff_buckets[1]++;
      }
      if (rel_diff < 0.1) {
        rel_diff_buckets[2]++;
      }
      if (rel_diff < 1) {
        rel_diff_buckets[3]++;
      } else {
        rel_diff_buckets[4]++;
      }
    }
  }

  spdlog::info("Difference Count:\n");
  spdlog::info("< 0.001: {} ({}%)\n", abs_diff_buckets[0],
               (float)abs_diff_buckets[0] / size * 100.0);
  spdlog::info("< 0.01: {} ({}%)\n", abs_diff_buckets[1],
               (float)abs_diff_buckets[1] / size * 100.0);
  spdlog::info("< 0.1: {} ({}%)\n", abs_diff_buckets[2],
               (float)abs_diff_buckets[2] / size * 100.0);
  spdlog::info("< 1: {} ({}%)\n", abs_diff_buckets[3],
               (float)abs_diff_buckets[3] / size * 100.0);
  spdlog::info("> 1: {} ({}%)\n", abs_diff_buckets[4],
               (float)abs_diff_buckets[4] / size * 100.0);

  spdlog::info("Percent Difference Count:\n");
  spdlog::info("< 0.001: {} ({}%)\n", rel_diff_buckets[0],
               (float)rel_diff_buckets[0] / size * 100.0);
  spdlog::info("< 0.01: {} ({}%)\n", rel_diff_buckets[1],
               (float)rel_diff_buckets[1] / size * 100.0);
  spdlog::info("< 0.1: {} ({}%)\n", rel_diff_buckets[2],
               (float)rel_diff_buckets[2] / size * 100.0);
  spdlog::info("< 1: {} ({}%)\n", rel_diff_buckets[3],
               (float)rel_diff_buckets[3] / size * 100.0);
  spdlog::info("> 1: {} ({}%)\n", rel_diff_buckets[4],
               (float)rel_diff_buckets[4] / size * 100.0);
  spdlog::info("\n");

  if (always_zero == 0.0) {
    spdlog::info("WARNING: All compared values are zero!\n");
  }

  // Ideally, these buckets should be non-overlapping...
  // TODO(fpedd): Subtract the next smaller bucket to make them non-overlapping
  float err = (1 - (float)rel_diff_buckets[1] / size) * 0.001 +
              (1 - (float)rel_diff_buckets[2] / size) * 0.01 +
              (1 - (float)rel_diff_buckets[3] / size) * 0.1 +
              (float)rel_diff_buckets[4] / size;
  return err * 100;
}