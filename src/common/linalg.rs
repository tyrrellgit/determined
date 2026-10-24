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
