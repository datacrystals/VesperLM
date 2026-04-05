#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting environment setup for JesperLM..."

# 1. Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "📦 Virtual environment already exists. Using existing 'venv'."
fi

source venv/bin/activate
echo "✅ Virtual environment activated."

# Upgrade pip to avoid build issues
pip install --upgrade pip setuptools wheel

# 2. Install PyTorch for ROCm 6.2
echo "🔥 Installing PyTorch (ROCm 6.2)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 3. Build bitsandbytes from source for MI50/MI60 (gfx906)
echo "🛠️ Cloning and building bitsandbytes for gfx906 architecture..."
if [ -d "bitsandbytes" ]; then
    echo "Cleaning up old bitsandbytes directory..."
    rm -rf bitsandbytes
fi

git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes

# Configure for HIP and explicitly target the MI50/MI60 architecture
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx906" -S .

# Compile using all available CPU cores
make -j$(nproc)

# Install into the venv
pip install .
cd ..
echo "✅ bitsandbytes built and installed."

# 4. Install remaining requirements
echo "📚 Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🎉 Setup complete! Run 'source venv/bin/activate' to get started."
python3 test.py
