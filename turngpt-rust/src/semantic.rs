/// Fast semantic arithmetic for turn vectors
/// Replaces torch.norm distance calculations that cause performance bottlenecks

/// Find closest turn vector using L2 distance
/// This replaces the torch distance calculation in semantic_arithmetic()
pub fn find_closest_turn_vector(
    target: &[i8], 
    vocab_turns: &[i8], 
    vocab_shape: &[usize]
) -> usize {
    let vocab_size = vocab_shape[0];
    let n_turns = vocab_shape[1];
    
    let mut min_distance = f32::INFINITY;
    let mut closest_idx = 0;
    
    for i in 0..vocab_size {
        let mut distance_squared = 0.0f32;
        
        // Calculate L2 distance squared (faster, no sqrt needed for comparison)
        for j in 0..n_turns {
            let vocab_idx = i * n_turns + j;
            let diff = target[j] as f32 - vocab_turns[vocab_idx] as f32;
            distance_squared += diff * diff;
        }
        
        if distance_squared < min_distance {
            min_distance = distance_squared;
            closest_idx = i;
        }
    }
    
    closest_idx
}

/// Perform semantic arithmetic: a - b + c
/// Returns the result turn vector
pub fn semantic_arithmetic(
    turn_a: &[i8], 
    turn_b: &[i8], 
    turn_c: &[i8]
) -> Vec<i8> {
    let n_turns = turn_a.len();
    let mut result = vec![0i8; n_turns];
    
    for i in 0..n_turns {
        // Clamp to i8 range to prevent overflow
        let val = turn_a[i] as i16 - turn_b[i] as i16 + turn_c[i] as i16;
        result[i] = val.clamp(-128, 127) as i8;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_arithmetic() {
        let a = vec![5, 3, -2, 1];
        let b = vec![2, 1, 0, -1];
        let c = vec![1, -1, 3, 2];
        
        let result = semantic_arithmetic(&a, &b, &c);
        // Expected: [5-2+1, 3-1-1, -2-0+3, 1-(-1)+2] = [4, 1, 1, 4]
        assert_eq!(result, vec![4, 1, 1, 4]);
    }
    
    #[test]
    fn test_closest_turn_vector() {
        let target = vec![1, 2];
        let vocab = vec![
            0, 0,  // distance = sqrt(5) ≈ 2.24
            1, 1,  // distance = sqrt(1) = 1
            2, 3,  // distance = sqrt(2) ≈ 1.41
        ];
        let vocab_shape = vec![3, 2];
        
        let closest = find_closest_turn_vector(&target, &vocab, &vocab_shape);
        assert_eq!(closest, 1); // [1, 1] is closest to [1, 2]
    }
}
