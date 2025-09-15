/// Fast polynomial evaluation for turn embeddings
/// Replaces the memory-intensive torch.einsum('bsp,po->bso', powers, poly_coeffs) operation

/// Evaluate polynomial for a batch of turn vectors
/// Simplified version that works correctly
pub fn evaluate_polynomial_batch(
    turns: &[i8],
    coeffs: &[f32], 
    turns_shape: &[usize],
    coeffs_shape: &[usize],
) -> Vec<f32> {
    let batch_size = turns_shape[0];
    let n_turns = turns_shape[1];
    let poly_degree = coeffs_shape[1] - 1; // coeffs_shape[1] = poly_degree + 1
    let output_dim = coeffs_shape[0]; // coeffs_shape[0] = output_dim
    
    let mut result = vec![0.0f32; batch_size * output_dim];
    
    // Process each batch item
    for batch_idx in 0..batch_size {
        // Process each turn dimension
        for turn_idx in 0..n_turns {
            let turn_value = turns[batch_idx * n_turns + turn_idx] as f32;
            
            // Generate polynomial powers: 1, x, x², x³, x⁴
            let mut power = 1.0f32;
            
            // Apply polynomial coefficients for this turn
            for d in 0..=poly_degree {
                for o in 0..output_dim {
                    let coeff_idx = d * output_dim + o;
                    let result_idx = batch_idx * output_dim + o;
                    
                    if coeff_idx < coeffs.len() && result_idx < result.len() {
                        result[result_idx] += power * coeffs[coeff_idx];
                    }
                }
                power *= turn_value;
            }
        }
    }
    
    result
}

/// Single turn polynomial evaluation (for testing)
pub fn evaluate_single_turn(turn_value: i8, coeffs: &[f32], poly_degree: usize) -> Vec<f32> {
    let output_dim = coeffs.len() / (poly_degree + 1);
    let mut result = vec![0.0f32; output_dim];
    let x = turn_value as f32;
    
    // Generate powers: 1, x, x², x³, x⁴
    let mut power = 1.0f32;
    
    for d in 0..=poly_degree {
        for o in 0..output_dim {
            let coeff_idx = d * output_dim + o;
            result[o] += power * coeffs[coeff_idx];
        }
        power *= x;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_turn_evaluation() {
        // Test with simple coefficients
        let coeffs = vec![1.0, 2.0, 0.5, 1.0]; // 2D output, degree 1
        let result = evaluate_single_turn(3, &coeffs, 1);
        
        // Expected: [1 + 3*0.5, 2 + 3*1] = [2.5, 5.0]
        assert!((result[0] - 2.5).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
    }
}
