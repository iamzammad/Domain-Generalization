import torch
from VIT_Model import load_vit_model

def verify_vit_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 10

    model = load_vit_model(num_classes, device)
    print(f"Model loaded successfully. Number of classes: {num_classes}")

    print("\nModel Architecture:")
    print(model)

    batch_size = 1 
    dummy_input = torch.randn(batch_size, 3, 112, 112).to(device)
    print(f"\nDummy input shape: {dummy_input.shape}")

    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")

        expected_shape = (batch_size, num_classes)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
        print("Output shape is correct.")

    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        return

    print("\nModel verification completed successfully!")

if __name__ == "__main__":
    verify_vit_model()
