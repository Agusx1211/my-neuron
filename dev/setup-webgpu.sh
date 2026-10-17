#!/bin/bash
# Run inside the isobox container as root: installs Chrome + NVIDIA Vulkan userland matching the host driver.
set -ex
VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
cd /tmp
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  [ -f NVIDIA-$VER.run ] || curl -fsSL -o NVIDIA-$VER.run https://us.download.nvidia.com/XFree86/Linux-x86_64/$VER/NVIDIA-Linux-x86_64-$VER.run
  rm -rf nv && sh NVIDIA-$VER.run --extract-only --target nv >/dev/null
  cd nv
  for f in libGLX_nvidia.so libnvidia-glcore.so libnvidia-glvkspirv.so libnvidia-gpucomp.so libnvidia-tls.so libEGL_nvidia.so libnvidia-eglcore.so libnvidia-glsi.so libnvidia-rtcore.so libnvidia-allocator.so; do
    [ -f $f.$VER ] && cp $f.$VER /usr/lib/x86_64-linux-gnu/ || echo "missing $f"
  done
  ln -sf libGLX_nvidia.so.$VER /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
  ln -sf libEGL_nvidia.so.$VER /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 || true
  ln -sf libnvidia-allocator.so.$VER /usr/lib/x86_64-linux-gnu/libnvidia-allocator.so.1 || true
  sed "s/__NV_VK_ICD__/libGLX_nvidia.so.0/" nvidia_icd.json.template > /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null || sed "s/__NV_VK_ICD__/libGLX_nvidia.so.0/" nvidia_icd.json > /usr/share/vulkan/icd.d/nvidia_icd.json
  ldconfig
  cd /tmp
fi
if ! command -v google-chrome >/dev/null; then
  curl -fsSL -o chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  apt-get update -qq && apt-get install -y -qq ./chrome.deb vulkan-tools libvulkan1 >/dev/null
fi
google-chrome --version
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverVersion|apiVersion" | head -6
