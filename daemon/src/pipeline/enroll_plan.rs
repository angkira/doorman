//! Which templates an enrolment run stores: near-duplicate pruning and the
//! merge / append policies. Pure functions over embeddings, no I/O.

use crate::ipc::select_diverse_embeddings;
use crate::ml::cosine_similarity;

/// How new templates combine with the ones already stored for the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnrollMode {
    /// Plain `enroll`: existing templates are never evicted. New distinct
    /// ones fill the free capacity, the most diverse of the new run's own
    /// templates winning when there are more than fit. A full gallery refuses
    /// new templates (`doorman remove <user>` starts over).
    Merge,
    /// `enroll --add`: same protection, but a run with no new distinct
    /// template is refused instead of being a no-op.
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
    if mode == EnrollMode::Append && distinct.is_empty() {
        return Err(
            "No new distinct templates: every captured face is a near-duplicate of a stored one. Try another pose or lighting."
                .to_string(),
        );
    }
    let capacity = target.saturating_sub(existing.len());
    if capacity == 0 && !distinct.is_empty() {
        return Err(format!(
            "Template gallery is full ({} of {target}); stored templates are never replaced. Nothing was stored. Run `doorman remove <user>` to start over.",
            existing.len()
        ));
    }
    let mut all = existing.to_vec();
    all.extend(select_diverse_embeddings(distinct, capacity));
    Ok(all)
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
        let kept = prune_near_duplicates(
            vec![at(5.0), at(40.0), at(42.0), at(90.0)],
            &existing,
            PRUNE,
        );
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
    fn merge_never_evicts_existing_templates_even_when_new_ones_are_outliers() {
        // The verifier's case: a full legacy gallery plus three clearly new faces.
        let existing: Vec<Vec<f32>> = (0..30).map(|i| at(i as f32 * 2.0)).collect();
        let full = plan_templates(
            vec![at(150.0), at(200.0), at(250.0)],
            &existing,
            EnrollMode::Merge,
            PRUNE,
            30,
        )
        .unwrap_err();
        assert!(
            full.contains("full") && full.contains("Nothing was stored"),
            "{full}"
        );
        // With 3 free slots all existing stay and the new ones fill them.
        let roomy = plan_templates(
            vec![at(150.0), at(200.0), at(250.0), at(300.0)],
            &existing[..27],
            EnrollMode::Merge,
            PRUNE,
            30,
        )
        .unwrap();
        assert_eq!(roomy.len(), 30);
        assert_eq!(&roomy[..27], &existing[..27]);
    }

    #[test]
    fn merge_with_nothing_new_keeps_the_stored_set_even_when_full() {
        let existing: Vec<Vec<f32>> = (0..3).map(|i| at(i as f32 * 90.0)).collect();
        let plan = plan_templates(vec![at(1.0)], &existing, EnrollMode::Merge, PRUNE, 3).unwrap();
        assert_eq!(plan, existing);
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
