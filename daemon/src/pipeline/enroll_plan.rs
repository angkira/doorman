//! Which templates an enrolment run stores: near-duplicate pruning and the
//! merge / append policies. Pure functions over embeddings, no I/O.

use crate::ipc::select_diverse_embeddings;
use crate::ml::cosine_similarity;

/// How new templates combine with the ones already stored for the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnrollMode {
    /// Plain `enroll`: union of old and new, then keep the `target` most
    /// diverse (this may evict older templates).
    Merge,
    /// `enroll --add`: existing templates are never evicted; new distinct
    /// ones fill the remaining capacity.
    Append,
}

/// Drop candidates whose cosine to an already kept template (`existing` or an
/// earlier accepted candidate) is `>= prune_cosine`. Order-preserving, greedy.
pub fn prune_near_duplicates(
    candidates: Vec<Vec<f32>>,
    existing: &[Vec<f32>],
    prune_cosine: f32,
) -> Vec<Vec<f32>> {
    let mut kept: Vec<Vec<f32>> = Vec::new();
    for candidate in candidates {
        let duplicate = existing
            .iter()
            .chain(kept.iter())
            .any(|t| cosine_similarity(&candidate, t) >= prune_cosine);
        if !duplicate {
            kept.push(candidate);
        }
    }
    kept
}

/// The full template set to store, or an explicit reason why nothing can be added.
pub fn plan_templates(
    new: Vec<Vec<f32>>,
    existing: &[Vec<f32>],
    mode: EnrollMode,
    prune_cosine: f32,
    target: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let distinct = prune_near_duplicates(new, existing, prune_cosine);
    match mode {
        EnrollMode::Merge => {
            let mut pool = distinct;
            pool.extend(existing.iter().cloned());
            Ok(select_diverse_embeddings(pool, target))
        }
        EnrollMode::Append => {
            if distinct.is_empty() {
                return Err(
                    "No new distinct templates: every captured face is a near-duplicate of a stored one. Try another pose or lighting."
                        .to_string(),
                );
            }
            let capacity = target.saturating_sub(existing.len());
            if capacity == 0 {
                return Err(format!(
                    "Template gallery is full ({} of {target}); remove the user to start over.",
                    existing.len()
                ));
            }
            let mut all = existing.to_vec();
            all.extend(select_diverse_embeddings(distinct, capacity));
            Ok(all)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(deg: f32) -> Vec<f32> {
        let r = deg.to_radians();
        vec![r.cos(), r.sin()]
    }

    const PRUNE: f32 = 0.95; // cos(18.2 degrees)

    #[test]
    fn prune_drops_near_duplicates_of_existing_and_of_earlier_candidates() {
        let existing = vec![at(0.0)];
        let kept =
            prune_near_duplicates(vec![at(5.0), at(40.0), at(42.0), at(90.0)], &existing, PRUNE);
        assert_eq!(kept, vec![at(40.0), at(90.0)]);
    }

    #[test]
    fn prune_drops_an_identical_vector_even_at_a_threshold_of_one() {
        let kept = prune_near_duplicates(vec![at(0.0)], &[at(0.0)], 1.0 - 1e-6);
        assert!(kept.is_empty());
    }

    #[test]
    fn merge_unions_and_keeps_at_most_target() {
        let existing = vec![at(0.0), at(60.0)];
        let plan = plan_templates(
            vec![at(120.0), at(2.0)],
            &existing,
            EnrollMode::Merge,
            PRUNE,
            30,
        )
        .unwrap();
        assert_eq!(plan.len(), 3, "the near-duplicate of at(0) is pruned");
        let capped = plan_templates(
            vec![at(120.0), at(180.0)],
            &existing,
            EnrollMode::Merge,
            PRUNE,
            3,
        )
        .unwrap();
        assert_eq!(capped.len(), 3);
    }

    #[test]
    fn append_never_evicts_existing_templates() {
        let existing = vec![at(0.0), at(60.0)];
        let plan = plan_templates(
            vec![at(120.0), at(180.0), at(240.0)],
            &existing,
            EnrollMode::Append,
            PRUNE,
            3,
        )
        .unwrap();
        assert_eq!(&plan[..2], &existing[..]);
        assert_eq!(plan.len(), 3, "only one slot was free");
    }

    #[test]
    fn append_reports_duplicates_and_a_full_gallery() {
        let existing = vec![at(0.0)];
        let dup =
            plan_templates(vec![at(3.0)], &existing, EnrollMode::Append, PRUNE, 30).unwrap_err();
        assert!(dup.contains("near-duplicate"), "{dup}");
        let full =
            plan_templates(vec![at(90.0)], &existing, EnrollMode::Append, PRUNE, 1).unwrap_err();
        assert!(full.contains("full"), "{full}");
    }
}
