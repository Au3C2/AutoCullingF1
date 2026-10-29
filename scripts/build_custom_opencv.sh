#!/usr/bin/env bash
# scripts/build_custom_opencv.sh — Build minimal standalone OpenCV wheel for AutoCulling.
#
# Strips FFmpeg, OpenEXR, GUI, GStreamer, and unused modules to eliminate 99 bundled
# dylibs (78 MB), producing a self-contained ~16 MB cv2.abi3.so linking only system dylibs.
# Used for P2 size optimization.
set -euo pipefail

OPENCV_VER="5.0.0"
OPENCV_PYTHON_TAG="93" # corresponds to 5.0.0.93
BUILD_DIR="${TMPDIR:-/tmp}/opencv_build"

echo "=== Building custom minimal OpenCV ${OPENCV_VER} (tag ${OPENCV_PYTHON_TAG}) ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [ ! -d "opencv-python" ]; then
    git clone --depth 1 --branch "${OPENCV_PYTHON_TAG}" https://github.com/opencv/opencv-python.git opencv-python
fi

cd opencv-python
git submodule update --init --depth 1 opencv

export CMAKE_ARGS="-DWITH_FFMPEG=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENCL=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_LIST=core,imgproc,imgcodecs,python3 \
  -DWITH_TIFF=ON \
  -DWITH_JPEG=ON \
  -DWITH_WEBP=ON \
  -DWITH_PNG=ON \
  -DCMAKE_BUILD_TYPE=Release"

echo "Running build..."
python setup.py bdist_wheel

echo "Build complete. Artifacts in dist/:"
ls -lh dist/*.whl
