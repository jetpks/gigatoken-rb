use icu::properties::props::{EnumeratedProperty, GeneralCategory, GeneralCategoryGroup};

#[inline]
pub(crate) fn get_general_category(c: char) -> GeneralCategory {
    GeneralCategory::for_char(c)
}

#[inline]
pub(crate) fn is_gc_letter(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Letter.contains(gc)
}

#[inline]
pub(crate) fn is_gc_number(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Number.contains(gc)
}

#[inline]
pub(crate) fn is_gc_separator(gc: GeneralCategory) -> bool {
    GeneralCategoryGroup::Separator.contains(gc)
}

#[inline]
pub(crate) fn is_letter(c: char) -> bool {
    is_gc_letter(get_general_category(c))
}

#[inline]
pub(crate) fn is_number(c: char) -> bool {
    is_gc_number(get_general_category(c))
}

#[inline]
pub(crate) fn is_separator(c: char) -> bool {
    is_gc_separator(get_general_category(c))
}

#[inline]
pub(crate) fn is_letter_complete(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    is_letter(c)
}

#[inline]
pub(crate) fn is_number_complete(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    is_number(c)
}

#[inline]
pub(crate) fn is_separator_complete(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_whitespace();
    }
    is_separator(c)
}

#[inline]
pub(crate) fn is_other_complete(c: char) -> bool {
    if c.is_ascii() {
        return !c.is_ascii_alphanumeric() && !c.is_ascii_whitespace();
    }
    let gc = get_general_category(c);
    !is_gc_letter(gc) && !is_gc_number(gc) && !is_gc_separator(gc)
}
