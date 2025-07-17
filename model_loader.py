# model_loader.py

import torch
import pickle
import os
from pathlib import Path


class ModelLoader:
    """Handles loading of PyTorch checkpoints in various formats"""

    def __init__(self, device='cpu'):
        self.device = device

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint from various formats:
        - Standard .pth files
        - Distributed format with data.pkl + data/ directory
        - Direct pickle files
        """
        checkpoint_path = Path(checkpoint_path)

        # Try different loading strategies
        strategies = [
            self._load_distributed_format,
            self._load_standard_pth,
            self._load_direct_pickle
        ]

        for strategy in strategies:
            try:
                print(f"Trying loading strategy: {strategy.__name__}")
                return strategy(checkpoint_path)
            except Exception as e:
                print(f"Strategy {strategy.__name__} failed: {e}")
                continue

        raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}")

    def _load_distributed_format(self, checkpoint_path):
        """Load checkpoint with data.pkl + data/ directory format"""
        if checkpoint_path.is_dir():
            pkl_file = checkpoint_path / 'data.pkl'
            data_dir = checkpoint_path / 'data'
        else:
            # If checkpoint_path points to data.pkl directly
            pkl_file = checkpoint_path
            data_dir = checkpoint_path.parent / 'data'

        if not pkl_file.exists() or not data_dir.exists():
            raise FileNotFoundError("Missing data.pkl or data/ directory")

        # Custom unpickler to handle persistent IDs
        class CheckpointUnpickler(pickle.Unpickler):
            def __init__(self, file, data_dir, device):
                super().__init__(file)
                self.data_dir = Path(data_dir)
                self.device = device

            def persistent_load(self, pid):
                """Handle persistent ID references to external tensor files"""
                if isinstance(pid, tuple) and len(pid) == 5:
                    # Format: ('storage', <class 'torch.FloatStorage'>, '0', 'cuda:0', 864)
                    typename, storage_class, data_id, original_device, size = pid
                    if typename == 'storage':
                        # Load tensor from external file
                        tensor_file = self.data_dir / str(data_id)
                        if tensor_file.exists():
                            # Read raw data
                            with open(tensor_file, 'rb') as f:
                                data = f.read()

                            # Create storage from raw bytes
                            storage = torch.UntypedStorage.from_buffer(
                                data, dtype=torch.uint8)

                            # Convert to typed storage and move to target device
                            if 'FloatStorage' in str(storage_class):
                                typed_storage = storage.float()
                            elif 'HalfStorage' in str(storage_class):
                                typed_storage = storage.half()
                            elif 'IntStorage' in str(storage_class):
                                typed_storage = storage.int()
                            elif 'LongStorage' in str(storage_class):
                                typed_storage = storage.long()
                            else:
                                typed_storage = storage.float()

                            # Move to target device
                            if self.device != 'cpu':
                                typed_storage = typed_storage.to(self.device)

                            return typed_storage

                # Fallback for unhandled persistent IDs
                raise pickle.UnpicklingError(
                    f"Unsupported persistent ID: {pid}")

        # Load with custom unpickler
        with open(pkl_file, 'rb') as f:
            unpickler = CheckpointUnpickler(f, data_dir, self.device)
            checkpoint = unpickler.load()

        return checkpoint

    def _load_standard_pth(self, checkpoint_path):
        """Load standard .pth checkpoint files"""
        if checkpoint_path.is_dir():
            # Look for .pth files in directory
            pth_files = list(checkpoint_path.glob('*.pth'))
            if not pth_files:
                raise FileNotFoundError("No .pth files found")
            checkpoint_path = pth_files[0]  # Use first .pth file

        return torch.load(checkpoint_path, map_location=self.device, weights_only=False)

    def _load_direct_pickle(self, checkpoint_path):
        """Load direct pickle files"""
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / 'data.pkl'

        # Try simple pickle load (may fail with persistent IDs)
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)


def load_model_checkpoint(checkpoint_path, device='cpu'):
    """
    Convenience function to load model checkpoint

    Args:
        checkpoint_path: Path to checkpoint (directory or file)
        device: Device to load tensors to ('cpu' or 'cuda')

    Returns:
        checkpoint: Loaded checkpoint state dict
    """
    loader = ModelLoader(device=device)
    return loader.load_checkpoint(checkpoint_path)


if __name__ == "__main__":
    # Test loading
    import sys
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'

        try:
            checkpoint = load_model_checkpoint(checkpoint_path, device)
            print(f"Successfully loaded checkpoint!")
            print(
                f"Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        print("Usage: python model_loader.py <checkpoint_path> [device]")
