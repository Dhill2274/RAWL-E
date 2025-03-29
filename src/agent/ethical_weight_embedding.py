import numpy as np
from scipy.spatial import ConvexHull

def get_hull(points):
    """

    Get_hull calculates the positive convex hull of a set of points, limiting it to only consider weights of the form
    (1, x, x) with x >= 0. If the number of points is too small to calculate the convex hull, the program will simply
    return the original points.

    :param points: set of 2-D points, they need to be numpy arrays
    :return: new set of 2-D points, the vertices of the calculated convex hull
    """
    try:
        hull = ConvexHull(points)
        vertices = []
        for vertex in hull.vertices:
            #print(points[vertex])
            vertices.append(points[vertex])

        vertices = np.array(vertices)

        best_individual = np.argmax(vertices[:, 0])

        #Calculating best ethical
        best_ethical = -1
        chosen_ethical = np.max(vertices[:, 1])

        where_ethical = np.argwhere(vertices[:, 1] == chosen_ethical)[:, 0]
        chosen_individual = np.max(vertices[where_ethical][:, 0])

        for i in range(len(vertices)):
            if vertices[i][0] == chosen_individual and vertices[i][1] == chosen_ethical:
                best_ethical = i

        #print(best_individual, best_ethical)

        if best_ethical < best_individual:
            vertices = np.concatenate((vertices[best_individual:], vertices[:best_ethical+1]),0)
        else:
            vertices = vertices[best_individual:best_ethical + 1]

        #print()
        #print(vertices)
        return vertices
    except:
        return points
    
def translate_hull(point, gamma, hull):
    if len(hull) == 0:
        hull = [point]
    else:
        for i in range(len(hull)):
            hull[i] = np.multiply(hull[i], gamma)
            hull[i] = np.add(hull[i], point)
    return hull
    
def compute_next_state(
    curr, 
    action_idx,
    reward,
):
    """
    A purely functional state transition for one agent:
      - Takes the current agent state (pos, health, berries, etc.)
      - Takes an integer action index (like 0=move, 1=eat, etc.)
      - Returns (next_state, reward_2d).

    This function must replicate the environment logic:
      1. "Move" updates pos if there's a path and sets a certain reward.
      2. "Eat" updates agent health and berries, sets a reward, etc.
      3. Then we update attributes (health decay, check if done, etc.).
    
    For multi-agent or more complex logic (finding nearest berry, etc.),
    you'd expand the logic here or pass in extra data.
    """

    # 1) Copy out fields from the current state
    health  = curr[0]
    berries = curr[1]
    day_to_live = curr[2]
    well_being = curr[3]
    dist_to_berry = curr[4]

    # If the agent is already done, we might just return the same state with zero reward
    if done:
        return (curr, np.array([0.0, 0.0], dtype=float))

    # 2) Interpret action
    reward_indiv = reward[1]
    reward_ethic = reward[0]

    # example mapping: 0=move, 1=eat, 2=throw, etc.
    if action_idx == 0:

        reward_vector = {0.0: "neutral",0.8: "forage"}
        # "move"
        new_x = dist_to_berry[0] - 1
        if reward_indiv >= 0:
            reward_indiv -= 1
        else:
            reward_indiv += 1
    elif action_idx == 1:

        reward_vector = {0.8: "eat", -0.1: "no_berries"}
        # "eat"
        if berries > 0:
            # +0.8 for eating, for example
            health += 0.6
            berries -= 1
            reward_indiv += 1.0  # your code uses "eat": 1.0
        else:
            reward_indiv += -0.2  # no berries penalty
    elif action_idx == 2:

        reward_vector = {0.5: "throw",-0.1: "no_benefactor"}
        # e.g. "throw" a berry
        if reward_indiv >= 0:
            reward_indiv -= 1
        else:
            reward_indiv += 1
    # else add more actions if needed

    # 3) Now do your "update_attributes" or "health_decay" logic
    # for example:
    health -= 0.1  # decay
    day_to_live = ((0.6 * berries) + health)/0.1

    if reward_ethic <= 0:
        done = True
        reward_indiv += -1.0  # "death" penalty
    else:
        day_to_live -= 1

    # 4) Combine into next_state
    next_state = np.array([health, berries, day_to_live, well_being, dist_to_berry], dtype=float)

    return next_state

def sample_based_partial_convex_hull_iteration(
    V: dict,
    replay_buffer: dict,
    q_network,
    discount_factor=0.95,
    max_iterations=7,
    batch_size=25
):
    """
    V: dict mapping from state -> np.array of shape (N,2)
    replay_buffer: your experience buffer with keys "s", "a", "r", "s_", "done"
    Each "r" is shape (2,) i.e. 2D reward.
    """
    if len(replay_buffer["s"]) < batch_size:
        print("Not enough experiences in buffer to do hull iteration.")
        return V
    
    for iteration in range(max_iterations):
        idxs = np.random.randint(0, len(replay_buffer["s"]), size=batch_size)
        for i in idxs:
            s      = np.asarray(replay_buffer["s"][i])  # or your ID for state
            done   = np.asarray(replay_buffer["done"][i])


            action_values = q_network.predict(np.atleast_2d(s.astype('float32')))[0]
            accum_points = []
            s_key = tuple(s)
            old_hull = V.get(s_key, np.zeros((0,2), dtype=np.float32))

            for index, action_reward in enumerate(action_values):
                # For each action, we get the future hull
                s_next = tuple(compute_next_state(s, index, action_reward))
                future_hull = np.zeros((0,2), dtype=np.float32)
                if not done:
                    future_hull = V.get(s_next, np.zeros((0,2), dtype=np.float32))

                new_points = translate_hull(action_reward, discount_factor, future_hull)
                accum_points.extend(new_points)
            
                combined   = np.concatenate([old_hull, new_points], axis=0)
                combined   = np.unique(combined, axis=0)
            V[s_key] = get_hull(combined)
        
        print(f"Iteration {iteration+1}/{max_iterations} done.")
    return V

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
        print("No points in hull.")
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
        print("Only one point in hull.")
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
                print("Infinite weight needed.")
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
                print("No weight needed.")
                return 0.0
            tie_weight = numerator / denom  # negative / negative => positive
            # To be strictly better, we must exceed tie_weight
            if tie_weight > max_weight:
                max_weight = tie_weight
                print("weight", max_weight)
    
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
