import torch
import math

def dof6_to_matrix(dof_actions: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 6-DOF actions (x, y, z, roll, pitch, yaw) to 4x4 homogeneous transformation matrices.
    Rotation order is ZYX (yaw, pitch, roll).
    
    Args:
        dof_actions: Tensor of shape (batch_size, 6)
                     Each row is [x, y, z, roll, pitch, yaw]
                     
    Returns:
        Tensor of shape (batch_size, 4, 4)
    """
    batch_size = dof_actions.shape[0]
    x, y, z, roll, pitch, yaw = dof_actions.unbind(dim=-1)

    # Precompute sin and cos
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)

    # Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = torch.zeros(batch_size, 3, 3, device=dof_actions.device, dtype=dof_actions.dtype)

    R[:, 0, 0] = cos_yaw * cos_pitch
    R[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    R[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll

    R[:, 1, 0] = sin_yaw * cos_pitch
    R[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll

    R[:, 2, 0] = -sin_pitch
    R[:, 2, 1] = cos_pitch * sin_roll
    R[:, 2, 2] = cos_pitch * cos_roll

    # Homogeneous transformation matrix
    T = torch.eye(4, device=dof_actions.device, dtype=dof_actions.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    T[:, :3, :3] = R
    T[:, 0, 3] = x
    T[:, 1, 3] = y
    T[:, 2, 3] = z
    
    return T

def matrix_to_dof6(matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 4x4 homogeneous transformation matrices to 6-DOF actions (x, y, z, roll, pitch, yaw).
    Assumes ZYX Euler angle convention (yaw, pitch, roll).
    
    Args:
        matrices: Tensor of shape (batch_size, 4, 4)
        
    Returns:
        Tensor of shape (batch_size, 6)
                     Each row is [x, y, z, roll, pitch, yaw]
    """
    R = matrices[:, :3, :3]
    
    # Translation
    x = matrices[:, 0, 3]
    y = matrices[:, 1, 3]
    z = matrices[:, 2, 3]

    # Rotation (Euler angles ZYX - yaw, pitch, roll)
    pitch = torch.atan2(-R[:, 2, 0], torch.sqrt(R[:, 2, 1]**2 + R[:, 2, 2]**2 + 1e-9)) # Added epsilon for stability

    # Threshold for gimbal lock
    epsilon = 1e-5 # Adjusted epsilon
    
    # Check for gimbal lock (cos(pitch) is close to 0, |sin(pitch)| is close to 1)
    # i.e., |R[2,0]| is close to 1
    not_gimbal_lock = torch.abs(R[:, 2, 0]) < 1.0 - epsilon
    gimbal_lock = ~not_gimbal_lock

    yaw = torch.zeros_like(pitch)
    roll = torch.zeros_like(pitch)

    # Non-gimbal lock case
    if torch.any(not_gimbal_lock):
        R_ng = R[not_gimbal_lock]
        pitch_ng = pitch[not_gimbal_lock]
        # cos_pitch_ng = torch.cos(pitch_ng) # This can be derived from R and pitch_ng
        # cos_pitch_ng = torch.sqrt(R_ng[:, 2, 1]**2 + R_ng[:, 2, 2]**2) -> This is sqrt( (c_p*s_r)^2 + (c_p*c_r)^2 ) = sqrt( c_p^2 * (s_r^2+c_r^2) ) = |c_p|
        # Using R[0,0] = c_y*c_p and R[1,0] = s_y*c_p. If c_p is not 0, can divide.
        # More robust: use atan2 with elements that don't become zero simultaneously with the denominator used for pitch.
        # R[0,0] = c_y*c_p, R[1,0] = s_y*c_p. yaw = atan2(s_y*c_p, c_y*c_p) = atan2(R[1,0], R[0,0])
        # R[2,1] = c_p*s_r, R[2,2] = c_p*c_r. roll = atan2(c_p*s_r, c_p*c_r) = atan2(R[2,1], R[2,2])

        yaw[not_gimbal_lock] = torch.atan2(R_ng[:, 1, 0], R_ng[:, 0, 0])
        roll[not_gimbal_lock] = torch.atan2(R_ng[:, 2, 1], R_ng[:, 2, 2])

    # Gimbal lock case
    if torch.any(gimbal_lock):
        R_g = R[gimbal_lock]
        # When pitch = +pi/2 (R[2,0] = -1), yaw - roll = atan2(R[0,1], R[0,2]) (using matrix elements from ZYX expansion)
        # When pitch = -pi/2 (R[2,0] =  1), yaw + roll = atan2(-R[0,1], -R[0,2]) (using matrix elements from ZYX expansion)
        # We set roll = 0 arbitrarily.
        roll[gimbal_lock] = 0.0
        
        # Check sign of R[2,0] to determine if pitch is +pi/2 or -pi/2
        # if R[2,0] is approx -1 (sin_pitch approx 1, pitch is +pi/2)
        # if R[2,0] is approx  1 (sin_pitch approx -1, pitch is -pi/2)
        
        # For pitch = +pi/2 (R[2,0] ~ -1): R[0,1] = -s_y*c_r + c_y*s_p*s_r = -s_y*c_r + c_y*s_r = sin(roll-yaw) if p=pi/2
        # R[1,1] = c_y*c_r + s_y*s_p*s_r = c_y*c_r + s_y*s_r = cos(roll-yaw) if p=pi/2
        # So, roll - yaw = atan2(R[0,1], R[1,1]). With roll=0, -yaw = atan2(R[0,1], R[1,1])
        # yaw = atan2(-R[0,1], R[1,1])
        
        # Let's use the common solution:
        # If R[2,0] = -1 (pitch = pi/2), then yaw = atan2(R[0,1], R[1,1]) (this assumes specific simplification for ZYX or similar)
        # If R[2,0] = 1 (pitch = -pi/2), then yaw = atan2(-R[0,1], R[1,1]) (this is one convention)
        # Alternative: yaw = atan2(R[1,2], R[0,2]) if R[2,0]=-1 (from some sources)
        # Let's use a derivation from R = Rz Ry Rx:
        # If pitch = pi/2 (sin_pitch=1, cos_pitch=0):
        # R[0,1] = cos_yaw * sin_roll - sin_yaw * cos_roll = sin(roll - yaw)
        # R[0,2] = cos_yaw * cos_roll + sin_yaw * sin_roll = cos(roll - yaw)
        # So roll - yaw = atan2(R[0,1], R[0,2]). If roll=0, yaw = -atan2(R[0,1],R[0,2]) = atan2(-R[0,1],R[0,2])
        # If pitch = -pi/2 (sin_pitch=-1, cos_pitch=0):
        # R[0,1] = -cos_yaw * sin_roll - sin_yaw * cos_roll = -sin(roll + yaw)
        # R[0,2] = -cos_yaw * cos_roll + sin_yaw * sin_roll = -cos(roll + yaw)
        # So roll + yaw = atan2(-R[0,1], -R[0,2]). If roll=0, yaw = atan2(-R[0,1],-R[0,2])

        idx_plus_pi_2 = torch.logical_and(gimbal_lock, R[:, 2, 0] < -1.0 + epsilon) # R[2,0] approx -1
        idx_minus_pi_2 = torch.logical_and(gimbal_lock, R[:, 2, 0] > 1.0 - epsilon) # R[2,0] approx 1
        
        if torch.any(idx_plus_pi_2):
             yaw[idx_plus_pi_2] = torch.atan2(-R[idx_plus_pi_2, 0, 1], R[idx_plus_pi_2, 0, 2])
        if torch.any(idx_minus_pi_2):
             yaw[idx_minus_pi_2] = torch.atan2(-R[idx_minus_pi_2, 0, 1], -R[idx_minus_pi_2, 0, 2]) # Note: atan2(A,B) == atan2(-A,-B) only if A,B are not both zero
                                                                                                # Check: atan2(X,Y) is angle of (Y,X) vector.
                                                                                                # if -A, -B then angle is same as A,B iff A,B not 0.
                                                                                                # My derivation was yaw = atan2(-R[0,1], -R[0,2]) which is fine.
                                                                                                
    return torch.stack([x, y, z, roll, pitch, yaw], dim=-1)

def matrix_inverse(matrices: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a batch of 4x4 homogeneous transformation matrices.
    
    For homogeneous transformation matrices T = [R t; 0 1], the inverse is:
    T^(-1) = [R^T -R^T*t; 0 1]
    
    Args:
        matrices: Tensor of shape (batch_size, 4, 4)
        
    Returns:
        Tensor of shape (batch_size, 4, 4) - inverted matrices
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    # Extract rotation matrix R (3x3) and translation vector t (3x1)
    R = matrices[:, :3, :3]  # [batch_size, 3, 3]
    t = matrices[:, :3, 3]   # [batch_size, 3]
    
    # Compute R^T (transpose of rotation matrix)
    R_transpose = R.transpose(-2, -1)  # [batch_size, 3, 3]
    
    # Compute -R^T * t
    t_inv = -torch.bmm(R_transpose, t.unsqueeze(-1)).squeeze(-1)  # [batch_size, 3]
    
    # Construct inverse matrix
    T_inv = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    T_inv[:, :3, :3] = R_transpose
    T_inv[:, :3, 3] = t_inv
    
    return T_inv

if __name__ == '__main__':
    # Test cases
    print("Testing transformation utils...")
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test 1: Identity
    actions_identity = torch.tensor([[0., 0., 0., 0., 0., 0.]], device=device, dtype=dtype)
    matrix_identity_expected = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    matrix_identity_actual = dof6_to_matrix(actions_identity)
    assert torch.allclose(matrix_identity_actual, matrix_identity_expected, atol=1e-6), "Test 1 failed: Identity matrix"
    actions_identity_recon = matrix_to_dof6(matrix_identity_actual)
    assert torch.allclose(actions_identity_recon, actions_identity, atol=1e-6), f"Test 1 failed: Identity recon. Got {actions_identity_recon}, expected {actions_identity}"
    print("Test 1 (Identity) passed.")

    # Test 2: Translation only
    actions_translation = torch.tensor([[1., 2., 3., 0., 0., 0.]], device=device, dtype=dtype)
    matrix_translation_actual = dof6_to_matrix(actions_translation)
    # Check T[:3,3] is [1,2,3]
    assert torch.allclose(matrix_translation_actual[0, :3, 3], actions_translation[0, :3], atol=1e-6), "Test 2 failed: Translation part of matrix"
    actions_translation_recon = matrix_to_dof6(matrix_translation_actual)
    assert torch.allclose(actions_translation_recon, actions_translation, atol=1e-6), f"Test 2 failed: Translation recon. Got {actions_translation_recon}, expected {actions_translation}"
    print("Test 2 (Translation) passed.")

    # Test 3: Rotation only (Yaw 90 deg)
    actions_yaw90 = torch.tensor([[0., 0., 0., 0., 0., math.pi/2]], device=device, dtype=dtype) # yaw = pi/2
    matrix_yaw90_actual = dof6_to_matrix(actions_yaw90)
    expected_R_yaw90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device, dtype=dtype)
    assert torch.allclose(matrix_yaw90_actual[0, :3, :3], expected_R_yaw90, atol=1e-6), "Test 3 failed: Yaw90 matrix"
    actions_yaw90_recon = matrix_to_dof6(matrix_yaw90_actual)
    assert torch.allclose(actions_yaw90_recon, actions_yaw90, atol=1e-6), f"Test 3 failed: Yaw90 recon. Got {actions_yaw90_recon}, expected {actions_yaw90}"
    print("Test 3 (Yaw 90 deg) passed.")

    # Test 4: Rotation only (Pitch 90 deg - Gimbal Lock)
    actions_pitch90 = torch.tensor([[0., 0., 0., 0., math.pi/2, 0.]], device=device, dtype=dtype) # pitch = pi/2
    matrix_pitch90_actual = dof6_to_matrix(actions_pitch90)
    expected_R_pitch90 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], device=device, dtype=dtype) # Rz(0)Ry(pi/2)Rx(0)
    assert torch.allclose(matrix_pitch90_actual[0, :3, :3], expected_R_pitch90, atol=1e-6), "Test 4 failed: Pitch90 matrix"
    
    actions_pitch90_recon = matrix_to_dof6(matrix_pitch90_actual)
    # Expected: x,y,z=0. pitch=pi/2. roll_recon=0. yaw_recon=0 (since original roll=0, yaw=0)
    # roll - yaw = atan2(R[0,1],R[0,2]) -> 0 - 0 = atan2(0,1) = 0. This is correct for pitch=pi/2 case where original yaw=0, roll=0
    
    # Print actual values for debugging
    print(f"  Pitch90 recon values: {actions_pitch90_recon}")
    print(f"  Expected pitch: {math.pi/2}")
    
    # Check individual components with appropriate tolerances
    assert torch.allclose(actions_pitch90_recon[:, :3], torch.zeros(1, 3, device=device, dtype=dtype), atol=1e-5), "Test 4 failed: xyz should be zero"
    assert torch.allclose(actions_pitch90_recon[:, 3], torch.zeros(1, device=device, dtype=dtype), atol=1e-5), "Test 4 failed: roll should be zero"
    assert torch.allclose(actions_pitch90_recon[:, 4], torch.tensor([math.pi/2], device=device, dtype=dtype), atol=1e-4), "Test 4 failed: pitch should be pi/2"
    assert torch.abs(actions_pitch90_recon[:, 5]).item() < 1e-4, "Test 4 failed: yaw should be approximately zero"
    print("Test 4 (Pitch 90 deg - Gimbal Lock) passed.")

    # Test 5: Combined transformation
    actions_combined = torch.tensor([[0.1, 0.2, 0.3, math.pi/6, math.pi/4, math.pi/3]], device=device, dtype=dtype) # r, p, y
    matrix_combined_actual = dof6_to_matrix(actions_combined)
    actions_combined_recon = matrix_to_dof6(matrix_combined_actual)
    assert torch.allclose(actions_combined_recon, actions_combined, atol=1e-5), f"Test 5 failed: Combined recon. Diff: {actions_combined_recon - actions_combined}"
    print("Test 5 (Combined) passed.")
    
    # Test 6: Batch processing & More Gimbal Lock
    # Case: pitch = pi/2, yaw = pi/4, roll = 0
    # R[2,0] = -sin(pi/2) = -1
    # Recon: x,y,z=0, pitch=pi/2, roll_recon=0
    # yaw_recon = -atan2(R[0,1], R[0,2])
    # R[0,1] = c_y*s_p*s_r - s_y*c_r = c(pi/4)*s(pi/2)*s(0) - s(pi/4)*c(0) = -sin(pi/4)
    # R[0,2] = c_y*s_p*c_r + s_y*s_r = c(pi/4)*s(pi/2)*c(0) + s(pi/4)*s(0) = cos(pi/4)
    # yaw_recon = -atan2(-sin(pi/4), cos(pi/4)) = -(-pi/4) = pi/4. Correct.
    
    # Case: pitch = -pi/2, yaw = pi/3, roll = pi/6
    # R[2,0] = -sin(-pi/2) = 1
    # Recon: x,y,z, pitch=-pi/2, roll_recon=0
    # yaw_recon = atan2(-R[0,1], -R[0,2])
    # R[0,1] = c_y*s_p*s_r - s_y*c_r = c(pi/3)*s(-pi/2)*s(pi/6) - s(pi/3)*c(pi/6)
    #        = c(pi/3)*(-1)*s(pi/6) - s(pi/3)*c(pi/6) = - (c(pi/3)s(pi/6) + s(pi/3)c(pi/6)) = -sin(pi/3+pi/6) = -sin(pi/2) = -1
    # R[0,2] = c_y*s_p*c_r + s_y*s_r = c(pi/3)*s(-pi/2)*c(pi/6) + s(pi/3)*s(pi/6)
    #        = c(pi/3)*(-1)*c(pi/6) + s(pi/3)*s(pi/6) = s(pi/3)s(pi/6) - c(pi/3)c(pi/6) = - (c(pi/3)c(pi/6) - s(pi/3)s(pi/6)) = -cos(pi/3+pi/6) = -cos(pi/2) = 0
    # yaw_recon = atan2(-(-1), -(0)) = atan2(1,0) = pi/2.
    # Original yaw was pi/3. This is because roll was non-zero. Our gimbal lock solution sets roll=0, which changes what yaw means.
    # The (yaw, roll) pair is (pi/3, pi/6). yaw+roll = pi/2 for pitch = -pi/2.
    # Recon (yaw_r=pi/2, roll_r=0). yaw_r+roll_r = pi/2. Consistent under ambiguity.

    actions_batch = torch.tensor([
        [0., 0., 0., 0., 0., 0.], # Identity
        [1., 2., 3., 0., 0., 0.], # Translation
        [0., 0., 0., 0., 0., math.pi/2], # Yaw
        [0., 0., 0., 0., math.pi/2, math.pi/4], # Gimbal lock: p=pi/2, y=pi/4, r=0
        [0., 0., 0., math.pi/6, -math.pi/2, math.pi/3], # Gimbal lock: p=-pi/2, y=pi/3, r=pi/6
        [0.1, 0.2, 0.3, math.pi/6, math.pi/4, math.pi/3] # Combined
    ], device=device, dtype=dtype)
    
    matrix_batch_actual = dof6_to_matrix(actions_batch)
    assert matrix_batch_actual.shape == (actions_batch.shape[0], 4, 4), "Test 6 failed: Batch matrix shape"
    
    actions_batch_recon = matrix_to_dof6(matrix_batch_actual)
    assert actions_batch_recon.shape == actions_batch.shape, "Test 6 failed: Batch recon shape"

    # Check each case. For gimbal lock, only x,y,z and pitch are guaranteed to be perfectly reconstructed
    # The sum/difference of yaw and roll should be preserved if the other is set to 0.
    # Our matrix_to_dof6 sets recon_roll=0 in gimbal lock.
    
    # Identity
    assert torch.allclose(actions_batch_recon[0], actions_batch[0], atol=1e-6), f"Batch Id recon failed. Got {actions_batch_recon[0]}"
    # Translation
    assert torch.allclose(actions_batch_recon[1], actions_batch[1], atol=1e-6), f"Batch Trans recon failed. Got {actions_batch_recon[1]}"
    # Yaw
    assert torch.allclose(actions_batch_recon[2], actions_batch[2], atol=1e-6), f"Batch Yaw recon failed. Got {actions_batch_recon[2]}"
    # Combined non-gimbal
    assert torch.allclose(actions_batch_recon[5], actions_batch[5], atol=1e-5), f"Batch Combined recon failed. Got {actions_batch_recon[5]}"

    # Gimbal Case 1: p=pi/2, y=pi/4, r=0
    # Expected recon: x,y,z=0. pitch=pi/2. roll_recon=0. yaw_recon = pi/4
    gimbal1_orig = actions_batch[3]
    gimbal1_recon = actions_batch_recon[3]
    
    print(f"  Gimbal1 orig: {gimbal1_orig}")
    print(f"  Gimbal1 recon: {gimbal1_recon}")
    print(f"  Pitch difference: {abs(gimbal1_recon[4] - gimbal1_orig[4])}")
    
    assert torch.allclose(gimbal1_recon[:3], gimbal1_orig[:3], atol=1e-4), "Gimbal1 xyz failed"
    assert torch.allclose(gimbal1_recon[4], gimbal1_orig[4], atol=1e-4), "Gimbal1 pitch failed"
    assert torch.allclose(gimbal1_recon[3], torch.tensor(0.0, device=device, dtype=dtype), atol=1e-4), "Gimbal1 recon roll should be 0"
    # original roll-yaw = 0 - pi/4 = -pi/4
    # recon roll-yaw = 0 - recon_yaw. So recon_yaw = pi/4
    assert torch.allclose(gimbal1_recon[5], gimbal1_orig[5], atol=1e-4), "Gimbal1 yaw failed" # Original yaw should be recovered if original roll was 0
    
    # Gimbal Case 2: p=-pi/2, y=pi/3, r=pi/6
    # Original: roll=pi/6, pitch=-pi/2, yaw=pi/3. So R[2,0] = -sin(-pi/2) = 1
    # Recon: x,y,z=0, pitch=-pi/2, roll_recon=0.
    # Original yaw+roll = pi/3+pi/6 = pi/2
    # Recon yaw_recon + roll_recon = yaw_recon. So yaw_recon should be pi/2.
    gimbal2_orig = actions_batch[4]
    gimbal2_recon = actions_batch_recon[4]
    expected_gimbal2_yaw_recon = gimbal2_orig[5] + gimbal2_orig[3] # yaw + roll

    assert torch.allclose(gimbal2_recon[:3], gimbal2_orig[:3], atol=1e-4), "Gimbal2 xyz failed"
    assert torch.allclose(gimbal2_recon[4], gimbal2_orig[4], atol=1e-4), "Gimbal2 pitch failed"
    assert torch.allclose(gimbal2_recon[3], torch.tensor(0.0, device=device, dtype=dtype), atol=1e-4), "Gimbal2 recon roll should be 0"
    assert torch.allclose(gimbal2_recon[5], expected_gimbal2_yaw_recon, atol=1e-4), f"Gimbal2 yaw failed. Got {gimbal2_recon[5]}, expected {expected_gimbal2_yaw_recon}"

    print("Test 6 (Batch with Gimbal Lock) passed sections that can be robustly checked.")
    print("Note: For gimbal lock cases where original roll != 0, recon roll is set to 0, and recon yaw absorbs the rotation.")
    
    # Test 7: Matrix inverse function
    print("\nTesting matrix_inverse function...")
    
    # Test with identity matrix
    identity_matrix = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    identity_inv = matrix_inverse(identity_matrix)
    assert torch.allclose(identity_inv, identity_matrix, atol=1e-6), "Test 7a failed: Identity matrix inverse"
    print("Test 7a (Identity inverse) passed.")
    
    # Test with translation matrix
    translation_actions = torch.tensor([[1., 2., 3., 0., 0., 0.]], device=device, dtype=dtype)
    translation_matrix = dof6_to_matrix(translation_actions)
    translation_inv = matrix_inverse(translation_matrix)
    
    # The inverse should give T * T^(-1) = I
    should_be_identity = torch.matmul(translation_matrix, translation_inv)
    expected_identity = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    assert torch.allclose(should_be_identity, expected_identity, atol=1e-6), "Test 7b failed: Translation matrix inverse"
    print("Test 7b (Translation inverse) passed.")
    
    # Test with rotation matrix
    rotation_actions = torch.tensor([[0., 0., 0., math.pi/6, math.pi/4, math.pi/3]], device=device, dtype=dtype)
    rotation_matrix = dof6_to_matrix(rotation_actions)
    rotation_inv = matrix_inverse(rotation_matrix)
    
    # The inverse should give T * T^(-1) = I
    should_be_identity = torch.matmul(rotation_matrix, rotation_inv)
    assert torch.allclose(should_be_identity, expected_identity, atol=1e-5), "Test 7c failed: Rotation matrix inverse"
    print("Test 7c (Rotation inverse) passed.")
    
    # Test with combined transformation
    combined_actions = torch.tensor([[1., 2., 3., math.pi/6, math.pi/4, math.pi/3]], device=device, dtype=dtype)
    combined_matrix = dof6_to_matrix(combined_actions)
    combined_inv = matrix_inverse(combined_matrix)
    
    # The inverse should give T * T^(-1) = I
    should_be_identity = torch.matmul(combined_matrix, combined_inv)
    assert torch.allclose(should_be_identity, expected_identity, atol=1e-5), "Test 7d failed: Combined matrix inverse"
    print("Test 7d (Combined inverse) passed.")
    
    # Test batch operation
    batch_actions = torch.tensor([
        [0., 0., 0., 0., 0., 0.],  # Identity
        [1., 2., 3., 0., 0., 0.],  # Translation
        [0., 0., 0., math.pi/4, 0., 0.],  # Rotation
        [0.5, 1.0, 1.5, math.pi/8, math.pi/6, math.pi/12]  # Combined
    ], device=device, dtype=dtype)
    
    batch_matrices = dof6_to_matrix(batch_actions)
    batch_inv = matrix_inverse(batch_matrices)
    
    # Test T * T^(-1) = I for all matrices in batch
    batch_identity_check = torch.matmul(batch_matrices, batch_inv)
    expected_batch_identity = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(4, 1, 1)
    assert torch.allclose(batch_identity_check, expected_batch_identity, atol=1e-5), "Test 7e failed: Batch matrix inverse"
    print("Test 7e (Batch inverse) passed.")
    
    print("All matrix_inverse tests passed!")
    
    print("All transformation utils tests passed where applicable!") 