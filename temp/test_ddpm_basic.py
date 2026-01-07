"""
Unit tests for DDPM implementation.

Run with: uv run python temp/test_ddpm_basic.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.ddpm import DDPM
from src.ddpm_solver import ddim_sample, ddpm_sample
from src.models import MLPVectorField, UNet


def test_beta_schedule():
    """Test that beta schedule is monotonically increasing and alphas are positive."""
    print("Testing beta schedule...")

    # Test with MLP
    model = MLPVectorField(hidden_dim=128, num_layers=3, use_discrete_time=True)
    ddpm = DDPM(model, num_timesteps=100)

    # Check betas are increasing
    assert torch.all(ddpm.betas[1:] >= ddpm.betas[:-1]), "Betas should be monotonically increasing"

    # Check alphas are positive
    assert torch.all(ddpm.alphas > 0), "Alphas should be positive"

    # Check alphas_cumprod are positive
    assert torch.all(ddpm.alphas_cumprod > 0), "Alpha_cumprod should be positive"

    # Check values are in reasonable range
    assert ddpm.betas[0] > 0 and ddpm.betas[0] < 0.1, "Beta_start should be small"
    assert ddpm.betas[-1] > 0 and ddpm.betas[-1] < 0.1, "Beta_end should be < 0.1"

    print("✅ Beta schedule test passed")


def test_loss_computation():
    """Test that DDPM loss computes correctly for 2D data."""
    print("\nTesting loss computation...")

    model = MLPVectorField(hidden_dim=128, num_layers=3, use_discrete_time=True)
    ddpm = DDPM(model, num_timesteps=100)

    # Test with 2D data
    x1 = torch.randn(32, 2)  # Batch of 32 2D samples
    loss = ddpm.compute_loss(x1)

    # Check loss is finite
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"

    # Check loss is positive
    assert loss.item() > 0, "Loss should be positive"

    # Check loss is scalar
    assert loss.dim() == 0, "Loss should be scalar"

    print(f"✅ Loss computation test passed (loss={loss.item():.6f})")


def test_forward_process():
    """Test forward diffusion process increases variance."""
    print("\nTesting forward process...")

    model = MLPVectorField(hidden_dim=128, num_layers=3, use_discrete_time=True)
    ddpm = DDPM(model, num_timesteps=100)

    x1 = torch.randn(16, 2)

    # Check variance at different timesteps using the q_sample method
    timesteps_to_test = [0, 10, 50, 99]
    variances = []

    for t in timesteps_to_test:
        t_batch = torch.full((16,), t, dtype=torch.long)
        epsilon = torch.randn_like(x1)

        # Use q_sample method
        x_t = ddpm.q_sample(x1, t_batch, epsilon)
        variances.append(x_t.var().item())

    # Variance should generally increase with t
    # (though there might be some fluctuation due to random noise)
    assert variances[-1] > variances[0], "Variance should increase at later timesteps"

    print(f"✅ Forward process test passed (variances: {[f'{v:.3f}' for v in variances]})")


def test_discrete_time_embedding():
    """Test that discrete time embedding normalizes correctly."""
    print("\nTesting discrete time embedding...")

    from src.models import DiscreteTimeEmbedding

    # Create embedding
    embed = DiscreteTimeEmbedding(dim=256, max_timesteps=1000)

    # Test with different timesteps
    t_early = torch.tensor([0])
    t_mid = torch.tensor([500])
    t_late = torch.tensor([999])

    emb_early = embed(t_early)
    emb_mid = embed(t_mid)
    emb_late = embed(t_late)

    # Check shapes
    assert emb_early.shape == (1, 256), f"Expected shape (1, 256), got {emb_early.shape}"
    assert emb_mid.shape == (1, 256), f"Expected shape (1, 256), got {emb_mid.shape}"
    assert emb_late.shape == (1, 256), f"Expected shape (1, 256), got {emb_late.shape}"

    # Check that embeddings are different for different timesteps
    assert not torch.allclose(emb_early, emb_mid), "Early and mid embeddings should differ"
    assert not torch.allclose(emb_mid, emb_late), "Mid and late embeddings should differ"

    print("✅ Discrete time embedding test passed")


def test_ddim_sampling_shapes():
    """Test that DDIM sampling produces correct shapes."""
    print("\nTesting DDIM sampling shapes...")

    model = MLPVectorField(hidden_dim=128, num_layers=3, use_discrete_time=True)

    # Test different NFE values
    nfe_values = [10, 20, 50]

    for nfe in nfe_values:
        samples = ddim_sample(
            model,
            num_samples=10,
            input_shape=(2,),
            num_steps=nfe,
            device="cpu",
        )

        # Check shape
        assert samples.shape == (10, 2), f"Expected (10, 2), got {samples.shape}"

        # Check samples are finite
        assert torch.all(torch.isfinite(samples)), "All samples should be finite"

    print(f"✅ DDIM sampling shapes test passed (tested NFE: {nfe_values})")


def test_model_with_discrete_time():
    """Test that models work correctly with discrete time embedding."""
    print("\nTesting models with discrete time...")

    # Test MLPVectorField
    model_mlp = MLPVectorField(
        hidden_dim=128,
        num_layers=3,
        use_discrete_time=True,
        max_timesteps=1000
    )

    x = torch.randn(4, 2)
    t = torch.tensor([0, 500, 999, 250])

    output_mlp = model_mlp(x, t)
    assert output_mlp.shape == (4, 2), f"Expected (4, 2), got {output_mlp.shape}"

    # Test UNet
    model_unet = UNet(
        model_channels=32,
        use_discrete_time=True,
        max_timesteps=1000
    )

    x_img = torch.randn(2, 3, 32, 32)
    t_img = torch.tensor([0, 500])

    output_unet = model_unet(x_img, t_img)
    assert output_unet.shape == (2, 3, 32, 32), f"Expected (2, 3, 32, 32), got {output_unet.shape}"

    print("✅ Model with discrete time test passed")


def test_cosine_beta_schedule():
    """Test cosine beta schedule."""
    print("\nTesting cosine beta schedule...")

    model = MLPVectorField(hidden_dim=128, num_layers=3, use_discrete_time=True)
    ddpm = DDPM(model, num_timesteps=100, beta_schedule="cosine")

    # Check basic properties
    assert torch.all(ddpm.betas > 0), "Cosine betas should be positive"
    assert torch.all(ddpm.betas < 1), "Cosine betas should be < 1"

    # Cosine schedule should start with small betas
    assert ddpm.betas[0] < 0.01, "Cosine schedule should start with small beta"

    print("✅ Cosine beta schedule test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running DDPM Unit Tests")
    print("=" * 60)

    try:
        test_beta_schedule()
        test_loss_computation()
        test_forward_process()
        test_discrete_time_embedding()
        test_ddim_sampling_shapes()
        test_model_with_discrete_time()
        test_cosine_beta_schedule()

        print("\n" + "=" * 60)
        print("✅ ALL UNIT TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
