//! Typed I32 and boolean operations over the shared strided word layout.
use super::elementwise::{PARAMETERS, SCALAR_RHS, validate_layout};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    F32,
    I32,
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum WordOp {
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Equal = 3,
    Greater = 4,
    GreaterEqual = 5,
    Less = 6,
    LessEqual = 7,
    Negate = 8,
    Absolute = 9,
    Sign = 10,
    BitAnd = 11,
    BitOr = 12,
    BitXor = 13,
    BitNot = 14,
    ShiftLeft = 15,
    ShiftRight = 16,
    BoolEqual = 17,
    BoolAnd = 18,
    BoolOr = 19,
    BoolXor = 20,
    BoolNot = 21,
    IntToFloat = 22,
    FloatToInt = 23,
    IntToBool = 24,
    FloatToBool = 25,
    BoolToFloat = 26,
    IntMaskFill = 27,
    BoolMaskFill = 28,
    Divide = 29,
    Remainder = 30,
}

impl WordOp {
    pub const ALL: [Self; 31] = [
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Equal,
        Self::Greater,
        Self::GreaterEqual,
        Self::Less,
        Self::LessEqual,
        Self::Negate,
        Self::Absolute,
        Self::Sign,
        Self::BitAnd,
        Self::BitOr,
        Self::BitXor,
        Self::BitNot,
        Self::ShiftLeft,
        Self::ShiftRight,
        Self::BoolEqual,
        Self::BoolAnd,
        Self::BoolOr,
        Self::BoolXor,
        Self::BoolNot,
        Self::IntToFloat,
        Self::FloatToInt,
        Self::IntToBool,
        Self::FloatToBool,
        Self::BoolToFloat,
        Self::IntMaskFill,
        Self::BoolMaskFill,
        Self::Divide,
        Self::Remainder,
    ];

    pub fn from_raw(raw: u32) -> Result<Self> {
        Self::ALL.get(raw as usize).copied().ok_or(Error::InvalidField)
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "burn_i32_add",
            Self::Subtract => "burn_i32_subtract",
            Self::Multiply => "burn_i32_multiply",
            Self::Equal => "burn_i32_equal",
            Self::Greater => "burn_i32_greater",
            Self::GreaterEqual => "burn_i32_greater_equal",
            Self::Less => "burn_i32_less",
            Self::LessEqual => "burn_i32_less_equal",
            Self::Negate => "burn_i32_negate",
            Self::Absolute => "burn_i32_absolute",
            Self::Sign => "burn_i32_sign",
            Self::BitAnd => "burn_i32_and",
            Self::BitOr => "burn_i32_or",
            Self::BitXor => "burn_i32_xor",
            Self::BitNot => "burn_i32_not",
            Self::ShiftLeft => "burn_i32_shift_left",
            Self::ShiftRight => "burn_i32_shift_right",
            Self::BoolEqual => "burn_bool_equal",
            Self::BoolAnd => "burn_bool_and",
            Self::BoolOr => "burn_bool_or",
            Self::BoolXor => "burn_bool_xor",
            Self::BoolNot => "burn_bool_not",
            Self::IntToFloat => "burn_i32_to_f32",
            Self::FloatToInt => "burn_f32_to_i32",
            Self::IntToBool => "burn_i32_to_bool",
            Self::FloatToBool => "burn_f32_to_bool",
            Self::BoolToFloat => "burn_bool_to_f32",
            Self::IntMaskFill => "burn_i32_mask_fill",
            Self::BoolMaskFill => "burn_bool_mask_fill",
            Self::Divide => "burn_i32_divide",
            Self::Remainder => "burn_i32_remainder",
        }
    }

    pub fn input(self) -> ValueType {
        match self {
            Self::FloatToInt | Self::FloatToBool => ValueType::F32,
            Self::BoolEqual | Self::BoolAnd | Self::BoolOr | Self::BoolXor | Self::BoolNot | Self::BoolToFloat | Self::BoolMaskFill => ValueType::Bool,
            _ => ValueType::I32,
        }
    }

    pub fn output(self) -> ValueType {
        match self {
            Self::FloatToInt => ValueType::I32,
            Self::IntToFloat | Self::BoolToFloat => ValueType::F32,
            Self::Equal | Self::Greater | Self::GreaterEqual | Self::Less | Self::LessEqual | Self::IntToBool | Self::FloatToBool => ValueType::Bool,
            _ => self.input(),
        }
    }

    pub fn mask_fill(self) -> bool {
        matches!(self, Self::IntMaskFill | Self::BoolMaskFill)
    }

    pub fn binary(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Subtract
                | Self::Multiply
                | Self::Divide
                | Self::Remainder
                | Self::Equal
                | Self::Greater
                | Self::GreaterEqual
                | Self::Less
                | Self::LessEqual
                | Self::BitAnd
                | Self::BitOr
                | Self::BitXor
                | Self::ShiftLeft
                | Self::ShiftRight
                | Self::BoolEqual
                | Self::BoolAnd
                | Self::BoolOr
                | Self::BoolXor
        )
    }

    pub fn unary(self) -> bool {
        !self.binary() && !self.mask_fill()
    }
}

pub fn validate(parameters: &[u32]) -> Result<[u64; 3]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let op = WordOp::from_raw(p[2] & 0xff)?;
    let scalar = p[2] & SCALAR_RHS != 0;
    if p[2] & !(SCALAR_RHS | 0xff) != 0
        || (scalar && !op.binary())
        || (!scalar && !op.mask_fill() && p[3] != 0)
        || ((scalar || op.mask_fill()) && op.input() == ValueType::Bool && p[3] > 1)
    {
        return Err(Error::InvalidField);
    }
    validate_layout(p)
}
