use crate::common::na as na;

pub fn moore_penrose_right_inverse<T, const R: usize, const C: usize>(
    a: &na::SMatrix<T, R, C>,
) -> Option<na::SMatrix<T, C, R>>
where
    T: nalgebra::RealField,
{
    // Moore-Penrose right inverse: A^+ = (A^T A)^-1 A^T
    let a_t = a.transpose();
    let ata = &a_t * a;
    let ata_inv = ata.try_inverse()?;
    Some(&ata_inv * &a_t)
}