# Copyright (c) Meta Platforms, Inc. and affiliates.
# V-JEPA 2-AC World Model Demo

import sys
sys.path.insert(0, '/home/ubuntu/vjepa2')

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from app.vjepa_droid.transforms import make_transforms


def poses_to_diff(pose1, pose2):
    """Compute the difference between two poses (simplified)."""
    return pose2 - pose1


def compute_new_pose(current_pose, action):
    """Compute new pose by adding action to current pose."""
    return current_pose + action


def main():
    print("=" * 60)
    print("V-JEPA 2-AC World Model Demo")
    print("=" * 60)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load V-JEPA 2-AC model
    print("\nLoading V-JEPA 2-AC model from PyTorch Hub...")
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    print(f"Encoder loaded: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M parameters")
    print(f"Predictor loaded: {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M parameters")

    # Setup transform
    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
    print(f"Tokens per frame: {tokens_per_frame}")

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    # Load robot trajectory
    print("\nLoading robot trajectory data...")
    trajectory = np.load("/home/ubuntu/vjepa2/notebooks/franka_example_traj.npz")
    np_clips = trajectory["observations"]  # (1, 2, 256, 256, 3)
    np_states = trajectory["states"]       # (1, 2, 7)

    print(f"Observations shape: {np_clips.shape}")
    print(f"States shape: {np_states.shape}")

    # Compute action (difference between states)
    np_actions = np.expand_dims(poses_to_diff(np_states[0, 0], np_states[0, 1]), axis=(0, 1))
    print(f"Action (delta pose): {np_actions[0, 0]}")

    # Convert to tensors
    clips = transform(np_clips[0]).unsqueeze(0).to(device)  # (1, C, T, H, W)
    states = torch.tensor(np_states, device=device, dtype=torch.float32)
    actions = torch.tensor(np_actions, device=device, dtype=torch.float32)

    print(f"\nInput tensors:")
    print(f"  clips: {clips.shape}")
    print(f"  states: {states.shape}")
    print(f"  actions: {actions.shape}")

    # Visualize the two frames
    print("\n" + "=" * 60)
    print("Visualizing trajectory frames...")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np_clips[0, 0])
    axes[0].set_title("Frame 1 (Current)")
    axes[0].axis('off')
    axes[1].imshow(np_clips[0, 1])
    axes[1].set_title("Frame 2 (Target/Future)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig("/home/ubuntu/vjepa2/notebooks/trajectory_frames.png", dpi=100)
    print("Saved: trajectory_frames.png")

    # Encode frames
    print("\n" + "=" * 60)
    print("Running World Model Inference")
    print("=" * 60)

    def encode_frames(clips):
        """Encode video clips to latent representations."""
        B, C, T, H, W = clips.size()
        # Reshape: each frame becomes a 2-frame "video" (duplicated)
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*T, C, H, W)
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (B*T, C, 2, H, W)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)  # (B, T*N, D)
        h = F.layer_norm(h, (h.size(-1),))
        return h

    with torch.inference_mode():
        # Encode both frames
        print("\n1. Encoding frames...")
        h = encode_frames(clips)
        print(f"   Encoded features shape: {h.shape}")
        print(f"   - {h.shape[1]} total tokens ({h.shape[1] // 2} per frame)")
        print(f"   - {h.shape[2]} dimensions")

        # Split into current frame and target frame features
        h_current = h[:, :tokens_per_frame]  # First frame
        h_target = h[:, -tokens_per_frame:]  # Second frame (ground truth future)

        print(f"\n2. Current frame features: {h_current.shape}")
        print(f"   Target frame features: {h_target.shape}")

        # Use predictor to predict future frame from current + action
        print("\n3. Predicting future frame using World Model...")
        print(f"   Input: current frame + action + state")

        # Predictor expects: (z, actions, states)
        s_current = states[:, :1]  # Current state
        h_predicted = predictor(h_current, actions, s_current)
        h_predicted = h_predicted[:, -tokens_per_frame:]  # Get predicted tokens
        h_predicted = F.layer_norm(h_predicted, (h_predicted.size(-1),))

        print(f"   Predicted features shape: {h_predicted.shape}")

        # Compare prediction with ground truth
        print("\n4. Comparing prediction with ground truth...")

        # Compute prediction error
        prediction_error = torch.abs(h_predicted - h_target).mean()
        cosine_sim = F.cosine_similarity(
            h_predicted.flatten(1),
            h_target.flatten(1)
        ).mean()

        print(f"   Mean Absolute Error: {prediction_error.item():.4f}")
        print(f"   Cosine Similarity: {cosine_sim.item():.4f}")

    # Energy landscape visualization
    print("\n" + "=" * 60)
    print("Computing Energy Landscape")
    print("=" * 60)
    print("Testing different actions to see which one matches the target best...")

    # Sample a grid of actions (simplified 2D grid for x and z)
    nsamples = 7
    grid_size = 0.1

    action_grid = []
    for dx in np.linspace(-grid_size, grid_size, nsamples):
        for dz in np.linspace(-grid_size, grid_size, nsamples):
            # Keep other dimensions at ground truth values
            action = np_actions[0, 0].copy()
            action[0] = dx  # x
            action[2] = dz  # z
            action_grid.append(action)

    action_grid = torch.tensor(np.array(action_grid), device=device, dtype=torch.float32).unsqueeze(1)
    print(f"Testing {len(action_grid)} different actions...")

    with torch.inference_mode():
        # Expand current state for all action samples
        h_current_expanded = h_current.repeat(len(action_grid), 1, 1)
        s_current_expanded = s_current.repeat(len(action_grid), 1, 1)

        # Predict for all actions
        h_pred_all = predictor(h_current_expanded, action_grid, s_current_expanded)
        h_pred_all = h_pred_all[:, -tokens_per_frame:]
        h_pred_all = F.layer_norm(h_pred_all, (h_pred_all.size(-1),))

        # Compute energy (prediction error) for each action
        h_target_expanded = h_target.repeat(len(action_grid), 1, 1)
        energy = torch.abs(h_pred_all - h_target_expanded).mean(dim=[1, 2])

    # Reshape energy to grid
    energy_grid = energy.cpu().numpy().reshape(nsamples, nsamples)

    # Plot energy landscape
    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = np.linspace(-grid_size, grid_size, nsamples)
    z_vals = np.linspace(-grid_size, grid_size, nsamples)

    im = ax.imshow(energy_grid.T, origin='lower',
                   extent=[x_vals[0], x_vals[-1], z_vals[0], z_vals[-1]],
                   cmap='viridis', aspect='auto')

    # Mark ground truth action
    gt_x, gt_z = np_actions[0, 0, 0], np_actions[0, 0, 2]
    ax.scatter([gt_x], [gt_z], c='red', s=200, marker='*',
               label=f'Ground Truth ({gt_x:.3f}, {gt_z:.3f})')

    # Find minimum energy action
    min_idx = np.argmin(energy_grid)
    min_x_idx, min_z_idx = np.unravel_index(min_idx, energy_grid.shape)
    pred_x, pred_z = x_vals[min_x_idx], z_vals[min_z_idx]
    ax.scatter([pred_x], [pred_z], c='cyan', s=200, marker='o',
               label=f'Lowest Energy ({pred_x:.3f}, {pred_z:.3f})')

    ax.set_xlabel('Action Delta X')
    ax.set_ylabel('Action Delta Z')
    ax.set_title('Energy Landscape\n(Lower = Better match to target frame)')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Prediction Error')

    plt.tight_layout()
    plt.savefig("/home/ubuntu/vjepa2/notebooks/energy_landscape.png", dpi=100)
    print("\nSaved: energy_landscape.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: What the World Model Does")
    print("=" * 60)
    print("""
The V-JEPA 2-AC World Model:

1. ENCODES video frames into latent representations
   - Each frame -> 256 tokens x 1408 dimensions

2. PREDICTS future states given:
   - Current frame features
   - Robot action (7D: x, y, z, rotation, gripper)
   - Current robot state

3. PLANS by finding actions that minimize "energy"
   - Energy = difference between predicted future and goal
   - Low energy = action leads to desired outcome

Use cases:
- Robot manipulation (reach, grasp, pick-and-place)
- Goal-conditioned control (give goal image, find actions)
- Model Predictive Control (MPC)
""")

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
