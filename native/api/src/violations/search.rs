use std::{cmp::Reverse, collections::BTreeSet};

use unicode_normalization::UnicodeNormalization;

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "at", "in", "is", "of", "on", "or", "the", "to", "which", "了", "和", "在",
    "是", "的",
];

pub(super) fn patterns(query: &str) -> Vec<String> {
    let mut text = String::with_capacity(query.len());
    for character in query.nfkc() {
        match character {
            'ß' | 'ẞ' => text.push_str("ss"),
            'ς' => text.push('σ'),
            _ => text.extend(character.to_lowercase()),
        }
    }
    let mut terms: BTreeSet<_> = text
        .split(|character| !is_word(character))
        .filter(|term| !term.is_empty() && !STOP_WORDS.contains(term))
        .collect();
    let mut matches = Vec::new();
    for (alias, expansion) in SYNONYMS {
        for (start, _) in text.match_indices(alias) {
            let end = start + alias.len();
            let inside_word = (alias.as_bytes()[0].is_ascii_alphanumeric()
                && text[..start].chars().next_back().is_some_and(is_word))
                || (alias.as_bytes()[alias.len() - 1].is_ascii_alphanumeric()
                    && text[end..].chars().next().is_some_and(is_word));
            if !inside_word {
                matches.push((start, end, alias.chars().count(), expansion));
            }
        }
    }

    // Apply long phrases before overlapping shorter matches.
    matches.sort_unstable_by_key(|&(start, _, width, _)| (Reverse(width), start));
    let mut covered = vec![false; text.len()];
    for (start, end, _, expansion) in matches {
        if !covered[start..end].iter().any(|&used| used) {
            terms.extend(*expansion);
            covered[start..end].fill(true);
        }
    }

    terms.into_iter().map(|term| format!("%{term}%")).collect()
}

fn is_word(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

// Keep this table aligned with examples/violation_records/search_utils.py.
const SYNONYMS: &[(&str, &[&str])] = &[
    ("帽", &["helmet", "no_helmet", "warning_no_hardhat"]),
    ("帽子", &["helmet", "no_helmet", "warning_no_hardhat"]),
    ("hat", &["helmet", "no_helmet", "warning_no_hardhat"]),
    ("helmet", &["helmet", "no_helmet", "warning_no_hardhat"]),
    ("casque", &["helmet", "no_helmet", "warning_no_hardhat"]),
    ("無安全帽", &["warning_no_hardhat"]),
    ("未戴帽", &["warning_no_hardhat"]),
    ("未戴安全帽", &["warning_no_hardhat"]),
    ("沒有安全帽", &["warning_no_hardhat"]),
    ("缺安全帽", &["warning_no_hardhat"]),
    ("no hardhat", &["warning_no_hardhat"]),
    ("not wearing hardhat", &["warning_no_hardhat"]),
    ("背", &["safety_vest", "no_safety_vest", "vest"]),
    ("背心", &["safety_vest", "no_safety_vest", "vest"]),
    ("vest", &["safety_vest", "no_safety_vest", "vest"]),
    ("gilet", &["safety_vest", "no_safety_vest", "vest"]),
    ("無背", &["warning_no_safety_vest"]),
    ("無安全背心", &["warning_no_safety_vest"]),
    ("未穿背", &["warning_no_safety_vest"]),
    ("未穿安全背心", &["warning_no_safety_vest"]),
    ("沒有安全背心", &["warning_no_safety_vest"]),
    ("缺安全背心", &["warning_no_safety_vest"]),
    ("no vest", &["warning_no_safety_vest"]),
    ("no safety vest", &["warning_no_safety_vest"]),
    ("人", &["person"]),
    ("人員", &["person"]),
    ("person", &["person"]),
    ("personne", &["person"]),
    ("personnes", &["person"]),
    ("機具", &["machinery"]),
    ("machine", &["machinery"]),
    ("machinery", &["machinery"]),
    ("machinerie", &["machinery"]),
    ("車", &["vehicle"]),
    ("車輛", &["vehicle"]),
    ("vehicle", &["vehicle"]),
    ("voiture", &["vehicle"]),
    ("錐", &["safety_cone"]),
    ("安全錐", &["safety_cone"]),
    ("cone", &["safety_cone"]),
    ("口罩", &["mask"]),
    ("mask", &["mask"]),
    ("無口罩", &["no_mask"]),
    ("no mask", &["no_mask"]),
    ("nomask", &["no_mask"]),
    ("no_helmet", &["warning_no_hardhat"]),
    ("no_safety_helmet", &["warning_no_hardhat"]),
    ("no_safety_vest", &["warning_no_safety_vest"]),
    ("no_mask", &["warning_no_mask"]),
    ("受控", &["warning_people_in_controlled_area"]),
    ("受控區", &["warning_people_in_controlled_area"]),
    ("受控區域", &["warning_people_in_controlled_area"]),
    ("控制區", &["warning_people_in_controlled_area"]),
    ("控制區域", &["warning_people_in_controlled_area"]),
    ("controlled", &["warning_people_in_controlled_area"]),
    ("controlled area", &["warning_people_in_controlled_area"]),
    ("controlled zone", &["warning_people_in_controlled_area"]),
    ("進入受控", &["warning_people_in_controlled_area"]),
    ("進入控制區", &["warning_people_in_controlled_area"]),
    ("進入控制區域", &["warning_people_in_controlled_area"]),
    ("靠近機具", &["warning_close_to_machinery"]),
    ("接近機具", &["warning_close_to_machinery"]),
    ("機具太近", &["warning_close_to_machinery"]),
    ("close machinery", &["warning_close_to_machinery"]),
    ("close to machinery", &["warning_close_to_machinery"]),
    ("靠近車輛", &["warning_close_to_vehicle"]),
    ("接近車輛", &["warning_close_to_vehicle"]),
    ("車輛太近", &["warning_close_to_vehicle"]),
    ("close vehicle", &["warning_close_to_vehicle"]),
    ("close to vehicle", &["warning_close_to_vehicle"]),
    ("電線桿控制區", &["detect_in_utility_pole_restricted_area"]),
    ("電桿控制區", &["detect_in_utility_pole_restricted_area"]),
    (
        "utility pole restricted area",
        &["detect_in_utility_pole_restricted_area"],
    ),
    (
        "pole restricted area",
        &["detect_in_utility_pole_restricted_area"],
    ),
    (
        "enter utility pole area",
        &["detect_in_utility_pole_restricted_area"],
    ),
    (
        "進入電線桿控制區",
        &["detect_in_utility_pole_restricted_area"],
    ),
    ("near_vehicle", &["warning_close_to_vehicle"]),
    ("near_machinery", &["warning_close_to_machinery"]),
    ("進入管制區", &["warning_people_in_controlled_area"]),
    ("restricted_area", &["warning_people_in_controlled_area"]),
    (
        "進入電桿管制區",
        &["warning_people_in_utility_pole_controlled_area"],
    ),
    (
        "utility_pole_restricted_area",
        &["warning_people_in_utility_pole_controlled_area"],
    ),
    ("機具靠近電桿", &["detect_machinery_close_to_pole"]),
    (
        "machinery_close_to_pole",
        &["detect_machinery_close_to_pole"],
    ),
];

#[cfg(test)]
mod tests {
    use super::patterns;

    #[test]
    fn expands_multilingual_phrases_and_keeps_literal_or_tokens() {
        assert_eq!(
            patterns("今日未戴安全帽，ＣＡＳＱＵＥ"),
            [
                "%casque%",
                "%helmet%",
                "%no_helmet%",
                "%warning_no_hardhat%",
                "%今日未戴安全帽%"
            ],
        );
        assert_eq!(
            patterns("no safety vest and what"),
            [
                "%no%",
                "%safety%",
                "%vest%",
                "%warning_no_safety_vest%",
                "%what%"
            ],
        );
        assert_eq!(patterns("在 the"), Vec::<String>::new());
    }
}
