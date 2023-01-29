use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{AddAssign, Mul, MulAssign},
    rc::Rc,
};

use num::traits::{MulAdd, One, Zero};

#[derive(Clone)]
struct GradFn<'a, T: Clone>(&'a dyn Fn(&Rc<RefCell<Differentiable<'a, T>>>) -> Vec<T>);

impl<T: Clone> Debug for GradFn<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GradFn")
    }
}

/// Represents a differentiable value of a given type.
#[derive(Clone, Debug)]
struct Differentiable<'a, T: Clone> {
    /// The value of the differentiable.
    pub value: T,
    /// The gradient of the differentiable.
    pub gradient: T,
    /// The Differentiable values that were used to compute this Differentiable.
    children: Vec<Rc<RefCell<Differentiable<'a, T>>>>,
    /// The function to compute the gradient of the children.
    grad_fn: GradFn<'a, T>,
}

fn topology_sort<'a, T: Clone>(
    diff: &Rc<RefCell<Differentiable<'a, T>>>,
) -> Vec<Rc<RefCell<Differentiable<'a, T>>>> {
    let mut result = Vec::new();

    for child in diff.borrow().children.iter() {
        result.append(&mut topology_sort(child));
    }

    result.push(diff.clone());

    result
}

fn backward<'a, T: One + Clone + AddAssign>(differentiable: &Rc<RefCell<Differentiable<'a, T>>>) {
    differentiable.borrow_mut().gradient += T::one();

    let sorted = topology_sort(differentiable);

    for diff in sorted.iter().rev() {
        let children = (diff.borrow().grad_fn.0)(diff);
        for (child, grad) in diff.borrow().children.iter().zip(children.iter()) {
            child.borrow_mut().gradient += grad.clone();
        }
    }
}

impl<T: Clone + Zero> From<T> for Differentiable<'_, T> {
    fn from(value: T) -> Self {
        Differentiable {
            value,
            gradient: T::zero(),
            children: Vec::new(),
            grad_fn: GradFn(&|_| Vec::new()),
        }
    }
}

impl<'a, T: Clone + Zero + MulAssign + Mul<Output = T>> Mul<T> for Differentiable<'a, T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let new_rhs = Differentiable::from(rhs);

        self * new_rhs
    }
}

impl<'a, T: Clone + Zero + MulAssign + Mul<T, Output = T>> Mul<Differentiable<'a, T>>
    for Differentiable<'a, T>
{
    type Output = Self;

    fn mul(self, rhs: Differentiable<'a, T>) -> Self::Output {
        Differentiable {
            value: self.value.clone() * rhs.value.clone(),
            gradient: T::zero(),
            children: vec![Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))],
            grad_fn: GradFn(&|diff| {
                vec![
                    diff.borrow().value.clone() * diff.borrow().children[1].borrow().value.clone(),
                    diff.borrow().value.clone() * diff.borrow().children[0].borrow().value.clone(),
                ]
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        let result = Differentiable::from(2);
        assert_eq!(result.value, 2);
    }

    #[test]
    fn multiplication() {
        let left = Differentiable::from(2);
        let right = Differentiable::from(3);
        let result = left * right;

        backward(result);

        // Print
        println!("{:?}", result);
    }
}
