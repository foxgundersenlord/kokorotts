cmake_minimum_required(VERSION 3.18)
project(KokoroTTS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_SOURCE_DIR}/libtorch")
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
    message(STATUS "Auto-detected libtorch at ${CMAKE_PREFIX_PATH}")
endif()

# Pre-declare torch::cudart so cuda.cmake's include-guard fires immediately.
# This lets the CUDA-enabled libtorch configure successfully without a CUDA
# toolkit installed — we link only against libtorch_cpu (CPU-only build).
add_library(torch::cudart INTERFACE IMPORTED GLOBAL)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

find_path(ESPEAK_INCLUDE_DIR espeak-ng/speak_lib.h
    PATHS /usr/include /usr/local/include)
find_library(ESPEAK_LIBRARY
    NAMES espeak-ng espeak
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

if(ESPEAK_INCLUDE_DIR AND ESPEAK_LIBRARY)
    message(STATUS "Found espeak-ng: ${ESPEAK_LIBRARY}")
    set(HAS_ESPEAK ON)
else()
    message(WARNING "espeak-ng not found — OOV words will be skipped. Install libespeak-ng-dev.")
    set(HAS_ESPEAK OFF)
endif()

set(SOURCES
    src/main.cpp
    src/kokoro/kokoro_tts.cpp
    src/misaki/tokenizer.cpp
    src/misaki/text_normalizer.cpp
    src/misaki/cmudict.cpp
    src/misaki/g2p.cpp
    src/audio/wav_writer.cpp
)

add_executable(kokoro_tts ${SOURCES})

target_include_directories(kokoro_tts PRIVATE
    ${TORCH_INCLUDE_DIRS}
    src/
)

# Link against libtorch_cpu directly so we don't pull in libtorch_cuda.so
# and its transitive CUDA toolkit dependencies.  libtorch.so (${TORCH_LIBRARIES})
# depends on libtorch_cuda.so which requires cuBLAS, cuDNN, etc.
# For GPU inference, replace libtorch_cpu with ${TORCH_LIBRARIES} and install CUDA.
target_link_libraries(kokoro_tts
    torch_cpu
    c10
    Threads::Threads
)

# libtorch_cpu.so has DT_NEEDED entries for the CUDA runtime and CUPTI even in
# CPU mode. Satisfy the linker with the stub .so files shipped in libtorch/lib.
foreach(_stub libcudart.so.12 libcupti.so.12)
    set(_stub_path "${CMAKE_SOURCE_DIR}/libtorch/lib/${_stub}")
    if(EXISTS "${_stub_path}")
        target_link_libraries(kokoro_tts "${_stub_path}")
    endif()
endforeach()

if(HAS_ESPEAK)
    target_compile_definitions(kokoro_tts PRIVATE KOKORO_HAS_ESPEAK)
    target_include_directories(kokoro_tts PRIVATE ${ESPEAK_INCLUDE_DIR})
    target_link_libraries(kokoro_tts ${ESPEAK_LIBRARY})
endif()

# Ensure shared Torch libs are found at runtime
set_target_properties(kokoro_tts PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Suppress noisy Torch deprecation warnings
target_compile_options(kokoro_tts PRIVATE -w)
