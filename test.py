import torch
import bitsandbytes as bnb

def test_8bit_optimizer():
    print(f"PyTorch version: {torch.__version__}")
    print(f"BitsAndBytes version: {bnb.__version__}")
    print("-" * 40)
    
    # Check if PyTorch can see the AMD GPU
    if not torch.cuda.is_available():
        print("❌ Error: PyTorch cannot detect a GPU. Ensure your ROCm PyTorch build is correct.")
        return

    device = torch.device("cuda")
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(device)}")

    # 1. Create a simple dummy model
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    ).to(device)

    # 2. Initialize the 8-bit optimizer
    print("\nInitializing 8-bit AdamW optimizer...")
    try:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)
        print("✅ 8-bit Optimizer initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize optimizer. Error:\n{e}")
        return

    # 3. Create dummy data
    inputs = torch.randn(32, 128, device=device)
    targets = torch.randn(32, 10, device=device)
    loss_fn = torch.nn.MSELoss()

    # 4. Perform a training step
    print("\nRunning a forward and backward pass...")
    try:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        # This is where bitsandbytes does the heavy lifting
        optimizer.step() 
        
        print(f"✅ Step completed successfully! Dummy loss: {loss.item():.4f}")
        print("\n🎉 Success: Your 8-bit optimizer is working perfectly on ROCm!")
    except Exception as e:
        print(f"❌ Error during training step:\n{e}")

if __name__ == "__main__":
    test_8bit_optimizer()
