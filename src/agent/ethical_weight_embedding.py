import numpy as np

def get_ethical_weight_for_state(hull_points: np.ndarray) -> float:
    """
    Given a 2D array of shape (N,2) where each row is (R0, R1),
    find the single weight w >= 0 so that the 'most ethical' point
    strictly dominates all other points under the linear scalarization:
         R0 + w * R1.
    
    For simplicity, we define the 'most ethical' point as whichever
    has the largest R1 (and break ties by picking the one with largest R0).
    
    Returns:
        A nonnegative float w_needed.  If w_needed = 0.0,
        it usually means there's only one point or no way to force
        that point to dominate under a positive weight.
    """
    if hull_points is None or len(hull_points) == 0:
        # Edge case: no points
        return 0.0
    
    # 1) Sort the hull by R1 ascending
    sorted_by_r1 = hull_points[hull_points[:, 1].argsort()]
    
    # The best ethical point is the row with the highest R1.
    # If there's a tie in R1, pick the one with highest R0 among them.
    # So we look at the last row(s).
    top_r1 = sorted_by_r1[-1, 1]  # highest R1
    # Extract all hull points that share this top R1
    candidates = sorted_by_r1[sorted_by_r1[:, 1] == top_r1]
    # Among these, pick the highest R0
    best_idx = np.argmax(candidates[:, 0])
    best_ethical = candidates[best_idx]  # shape (2,)
    
    # If there's only one point, there's no conflict
    if len(hull_points) < 2:
        return 0.0
    
    # 2) Among the other hull points, find the weight that just ties best_ethical
    #    for each competitor, then pick the maximum of those (so that we exceed them).
    max_weight = 0.0
    
    # We'll define best_ethical = (be0, be1) for convenience
    be0, be1 = best_ethical
    for row in hull_points:
        # skip comparing best_ethical to itself
        if np.allclose(row, best_ethical):
            continue
        r0, r1 = row
        # We want: (be0 + w*be1) > (r0 + w*r1)
        # Rearrange: be0 - r0 > w * (r1 - be1)
        # => w < (be0 - r0) / (r1 - be1), if (r1 - be1) < 0
        # if r1 == be1 and be0 <= r0, then it's impossible to strictly dominate
        # if be0 - r0 <= 0, we can't strictly dominate that competitor at all.
        
        # We only care if be1 > r1. Otherwise if r1 == be1
        # but row has a bigger R0, no w can force a strict improvement.
        # If r1 - be1 < 0 => be1 > r1 => let's compute the tie weight:
        denom = (r1 - be1)
        numerator = (be0 - r0)
        
        if np.isclose(denom, 0.0):
            # If the competitor has same R1:
            #  - if be0 <= r0, can't strictly dominate => infinite needed
            #  - if be0 > r0, we don't need any w>0 to dominate it in R0 alone
            # => we skip unless be0 <= r0
            if np.isclose(be1, r1) and be0 <= r0:
                # can't strictly dominate under any w
                return 0.0
            else:
                # no special constraint needed
                continue
        else:
            # denom != 0
            # We only get a real constraint if be1 > r1 => denom < 0
            # and be0 > r0 => numerator > 0
            if denom >= 0 or numerator <= 0:
                # can't forcibly outscore this competitor with a finite w>0
                return 0.0
            tie_weight = numerator / denom  # negative / negative => positive
            # To be strictly better, we must exceed tie_weight
            if tie_weight > max_weight:
                max_weight = tie_weight
    
    return max_weight


def compute_global_ethical_weight(all_state_hulls: dict, epsilon: float) -> float:
    """
    For each state s in all_state_hulls, we compute the local weight
    that ensures the 'most ethical' point strictly dominates.  We then
    pick the *maximum* of these local weights.  Finally, we add an 
    epsilon>0 so that we ensure strict dominance across *all* states.
    
    Args:
      all_state_hulls: A dict { state : np.array of shape(N,2) }, 
                       representing the multi-objective returns for each state.
      epsilon: small positive number to ensure strictness.
    
    Returns:
      A single float w_global that ensures that in *every* state,
      the top ethical point is strictly optimal under R0 + w_global*R1.
    """
    best_so_far = 0.0
    for s, points in all_state_hulls.items():
        local_w = get_ethical_weight_for_state(points)
        if local_w > best_so_far:
            best_so_far = local_w
    
    return best_so_far + epsilon
