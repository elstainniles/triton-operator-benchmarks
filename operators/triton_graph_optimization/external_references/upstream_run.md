## chunk_scaled_dot_kkt_fwd_kernel
**必须要 Q2TritonKernel-autotune 依赖**
git clone https://gitcode.com/TritonAscendTest/Q2TritonKernel.git
cd ./Q2TritonKernel-autotune
TRITON_ALWAYS_COMPILE=1 PYTHONPATH=$PWD/src/kernels:$PWD pytest tests/fla/test_chunk_scaled_dot_kkt_fwd_perf.py -sv

## chunk_local_cumsum (fla)
**必须要 Q2TritonKernel-autotune 依赖**
git clone https://gitcode.com/TritonAscendTest/Q2TritonKernel.git
cd ../Q2TritonKernel-autotune
TRITON_ALWAYS_COMPILE=1 PYTHONPATH=$PWD/src/kernels:$PWD pytest tests/fla/test_chunk_local_cumsum_perf.py -sv
打开生成的10个csv文件，计算平均值

## matmul_kernel
**必须要 LigerKernel 依赖**
git clone https://gitcode.com/TritonAscendTest/LigerKernel.git
rm -rf cache && TRITON_CACHE_DIR=$(pwd)/cache TRITON_DEBUG=1 TRITON_ALWAYS_COMPILE=1 PYTHONPATH=$PWD/src/:$PWD python benchmark/scripts/benchmark_mm_int8int2.py