def compute_spatial_trajectory(height: int, width: int, num_stages: int) -> list[tuple[int, int]]:
    """Compute the spatial dimensions at each encoder pooling stage.
    
    Returns a list of (h, w) tuples, from input resolution down to the 
    bottleneck, with length num_stages + 1.
    
    Example for 224, 6 stages:
        [(224, 224), (112, 112), (56, 56), (28, 28), (14, 14), (7, 7), (3, 3)]
    """
    trajectory = [(height, width)]
    h, w = height, width
    for _ in range(num_stages):
        h, w = h // 2, w // 2
        trajectory.append((h, w))
    return trajectory


def compute_output_paddings(trajectory: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Compute the output_padding needed at each decoder upsampling stage.
    
    For each transition trajectory[i+1] -> trajectory[i], if 
    trajectory[i] was odd, Conv2DTranspose(stride=2) will produce 
    trajectory[i+1]*2 which is 1 less than trajectory[i], so we need 
    output_padding=1 on that axis.
    
    Returns paddings in decoder order (bottleneck -> full resolution),
    i.e., reversed from the encoder trajectory.
    
    Example for 224, 6 stages:
        trajectory: [(224,224), (112,112), (56,56), (28,28), (14,14), (7,7), (3,3)]
        decoder goes: 3->7->14->28->56->112->224
        output_paddings: [(1,1), (0,0), (0,0), (0,0), (0,0), (0,0)]
        (the first upsample 3*2=6, need 7, so pad 1)
    """
    paddings = []
    # Traverse from bottleneck back to input
    for i in range(len(trajectory) - 1, 0, -1):
        target_h, target_w = trajectory[i - 1]
        source_h, source_w = trajectory[i]
        pad_h = target_h - source_h * 2  # 0 or 1
        pad_w = target_w - source_w * 2  # 0 or 1
        paddings.append((pad_h, pad_w))
    return paddings