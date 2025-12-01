import sys
import torch
import torch.nn as nn
from types import SimpleNamespace

# Add path
sys.path.append('/workspace/REPAIR')
sys.path.append('/workspace/REPAIR/ELDER/peft_egg/src')

from peft.tuners.elder import ELDERModel, ElderLinear, ElderGraceLinear

def test_elder_sequential():
    print("Testing ELDER sequential execution...")
    
    # Mock Config
    grace_config = {'num_rank_per_block': 2, 'init_radius': 1.0, 'threshold': 0.5}
    
    # 1. Create ElderGraceLinear (The "Sensor")
    print("Creating ElderGraceLinear...")
    # It needs gate_linears. We'll create a dummy one.
    dummy_gate = nn.Linear(32, 4)
    
    grace_layer = ElderGraceLinear(
        adapter_name="default",
        in_features=32,
        out_features=32,
        grace_config=grace_config,
        gate_linears=[dummy_gate],
        top_k=2,
        num_experts=4
    )
    grace_layer.editing = False # Inference
    
    # 2. Create ElderLinear (The "Actor")
    print("Creating ElderLinear...")
    elder_linear = ElderLinear(
        adapter_name="default",
        in_features=32,
        out_features=32,
        r=8,
        num_experts=4,
        top_k=2
    )
    elder_linear.editing = False # Inference
    
    # 3. Run Forward Pass
    x = torch.randn(1, 10, 32)
    
    print("Running GraceLayer forward (should set IN_EDIT_SCOPE)...")
    # GraceLayer needs key_id. Default -1 means use input.
    # It sets SEQ_REPR and IN_EDIT_SCOPE.
    # But it needs to discriminate. Discriminate needs binary_code.
    # binary_code needs get_bin_code.
    # get_bin_code needs gate_linears.
    
    # We need to mock discriminate to avoid complex setup
    # But we want to test the global variable setting.
    
    # Let's mock get_bin_code to do nothing and set binary_code manually
    grace_layer.binary_code = torch.zeros(1, 4) 
    # And mock discriminate to return a list
    grace_layer.discriminate = lambda threshold: [False] * x.shape[0]
    
    out_grace = grace_layer(x)
    print("GraceLayer forward done.")
    
    # Check if IN_EDIT_SCOPE is set
    import peft.tuners.elder as elder_module
    if hasattr(elder_module, 'IN_EDIT_SCOPE'):
        print(f"Global IN_EDIT_SCOPE set: {elder_module.IN_EDIT_SCOPE}")
    else:
        print("Global IN_EDIT_SCOPE NOT set!")
        
    print("Running ElderLinear forward (should use IN_EDIT_SCOPE)...")
    # It uses SEQ_REPR. GraceLayer sets it.
    try:
        out_linear = elder_linear(x)
        print("ElderLinear forward successful!")
    except Exception as e:
        print(f"ElderLinear forward failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_elder_sequential()
