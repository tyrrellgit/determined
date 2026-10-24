use nalgebra as na;
use na::allocator::Allocator;

/// Moore-Penrose pseudo-inverse of a full-rank matrix.
///
/// Uses the left inverse `(A^T A)^-1 A^T` when `A` has full column rank (tall
/// or square), otherwise the right inverse `A^T (A A^T)^-1` when it has full
/// row rank (wide). Returns `None` if `A` is rank deficient.
pub fn moore_penrose_inverse<T, R, C>(a: &na::OMatrix<T, R, C>) -> Option<na::OMatrix<T, C, R>>
where
    T: na::RealField,
    R: na::Dim,
    C: na::Dim,
    na::DefaultAllocator: Allocator<R, C> + Allocator<C, R> + Allocator<C, C> + Allocator<R, R>,
{
    let a_t = a.transpose();
    if let Some(ata_inv) = (&a_t * a).try_inverse() {
        return Some(ata_inv * a_t);
    }
    let aat_inv = (a * &a_t).try_inverse()?;
    Some(a_t * aat_inv)
}

/// UD decomposition `M = U D U^T` of a symmetric positive semi-definite
/// matrix, with `U` unit upper triangular and `D` diagonal (returned as a
/// vector). Ramos et al., "A summary on the UD Kalman Filter" (arXiv:2203.06105),
/// Appendix A.
///
/// A zero pivot (a semi-definite `M`) gives zero off-diagonal entries in that
/// column of `U`. A negative entry in `D` means `M` was not positive
/// semi-definite.
pub fn udu_decomposition<D>(m: &na::OMatrix<f64, D, D>) -> (na::OMatrix<f64, D, D>, na::OVector<f64, D>)
where
    D: na::Dim,
    na::DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    let (nd, _) = m.shape_generic();
    let n = m.nrows();
    let mut u = na::OMatrix::<f64, D, D>::identity_generic(nd, nd);
    let mut d = na::OVector::<f64, D>::zeros_generic(nd, na::Const::<1>);

    // pivots below this are treated as zero
    let scale = m.diagonal().iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    let tiny = f64::EPSILON * scale.max(f64::MIN_POSITIVE);

    for j in (0..n).rev() {
        for i in (0..=j).rev() {
            let mut sigma = m[(i, j)];
            for k in j + 1..n {
                sigma -= u[(i, k)] * d[k] * u[(j, k)];
            }
            if i == j {
                d[j] = sigma;
            } else if d[j].abs() > tiny {
                u[(i, j)] = sigma / d[j];
            }
        }
    }
    (u, d)
}
