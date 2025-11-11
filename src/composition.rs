pub trait Composable<Rhs = Self> {
    type Output;
    fn add(self, other: Rhs) -> Self::Output;
};