import numpy as np
from shadow_manager import ShadowIndexManager

def test_shadow_dual_write_and_atomic_swap():
    manager = ShadowIndexManager(initial_model="all-MiniLM-L6-v2", dimension=4)
    v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    manager.write_live(v1, {"prompt": "Initial Live Cache Prompt"})

    
    manager.start_migration(target_model="bge-small-en-v1.5", target_dim=4, target_entries=2)
    
    v2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    v3 = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    manager.write_shadow(v2, {"prompt": "Warmup 1"})
    manager.write_shadow(v3, {"prompt": "Warmup 2"})

    status = manager.get_status()
    assert status.can_swap is True
    assert status.readiness_pct == 100.0

  
    swapped = manager.atomic_swap()
    assert swapped is True
    assert manager.live_model == "bge-small-en-v1.5"
    assert len(manager.live_metadata) == 2

    rolled_back = manager.rollback()
    assert rolled_back is True
    assert manager.live_model == "all-MiniLM-L6-v2"
    assert len(manager.live_metadata) == 1
