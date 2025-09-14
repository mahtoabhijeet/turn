use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, PyArray2};

mod polynomial;
mod semantic;

use polynomial::evaluate_polynomial_batch;
use semantic::find_closest_turn_vector;

/// PyO3 Python bindings for TurnGPT Rust acceleration
#[pymodule]
fn turngpt_rust(py: Python, m: &PyModule) -> PyResult<()> {
    
    /// Fast polynomial evaluation for turn embeddings
    /// Replaces the memory-intensive torch.einsum operation
    #[pyfn(m)]
    fn evaluate_turns(
        py: Python,
        turns: &PyArray2<i8>,
        coeffs: &PyArray2<f32>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let turns_slice = turns.readonly();
        let coeffs_slice = coeffs.readonly();
        
        let result = evaluate_polynomial_batch(
            turns_slice.as_slice()?,
            coeffs_slice.as_slice()?,
            turns.shape(),
            coeffs.shape(),
        );
        
        let output_shape = [turns.shape()[0], coeffs.shape()[1]];
        Ok(PyArray2::from_vec2(py, &vec![result])?.to_owned())
    }
    
    /// Fast semantic arithmetic for turn vectors
    /// Replaces torch distance calculations
    #[pyfn(m)]
    fn find_closest_turn(
        py: Python,
        target_turns: &PyArray1<i8>,
        vocab_turns: &PyArray2<i8>,
    ) -> PyResult<usize> {
        let target_slice = target_turns.readonly();
        let vocab_slice = vocab_turns.readonly();
        
        let closest_idx = find_closest_turn_vector(
            target_slice.as_slice()?,
            vocab_slice.as_slice()?,
            vocab_turns.shape(),
        );
        
        Ok(closest_idx)
    }

    Ok(())
}
