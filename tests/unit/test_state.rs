#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = State::new(1.0, 0);
        let b = State::new(2.0, 0);
        let result = a.add(b);
        // Replace with your actual logic
        assert_eq!(result.value, 3.0);
    }
}