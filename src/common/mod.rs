pub use nalgebra as na;

pub mod linalg;
pub mod composition;

pub use composition::Composable;

use na::allocator::Allocator;

/// nalgebra allocator bounds needed by the filter models, for a state of
/// dimension `N` and a measurement of dimension `M`.
///
/// Lets the models be written once for both statically sized (`na::Const<N>`)
/// and dynamically sized (`na::Dyn`) matrices. Use as
/// `where na::DefaultAllocator: FilterAllocator<N, M>`.
pub trait FilterAllocator<N: na::Dim, M: na::Dim>:
    Allocator<N> + Allocator<M>
    + Allocator<N, N> + Allocator<M, M>
    + Allocator<M, N> + Allocator<N, M>
{}

impl<N: na::Dim, M: na::Dim, A> FilterAllocator<N, M> for A
where
    A: Allocator<N> + Allocator<M>
        + Allocator<N, N> + Allocator<M, M>
        + Allocator<M, N> + Allocator<N, M>
{}
