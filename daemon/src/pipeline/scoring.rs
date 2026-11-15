//! Scoring one embedding against a user's stored templates.
//!
//! `max` is the comparison the daemon always did (cosine to every stored
//! template, keep the best), so it is the default; `top3_mean` and `centroid`
//! are opt-in via `[decision] template_score`. Pure functions, no I/O.

use super::aggregate_embeddings;
use crate::ml::cosine_similarity;
use doorman_shared::config::TemplateScore;

/// How many of the best template cosines `top3_mean` averages.
const TOP_K: usize = 3;

/// Score `embedding` against `templates`. An empty gallery scores 0.0, and for
/// `max` a negative best cosine is floored at 0.0 (as the original fold did).
pub fn template_score(mode: TemplateScore, embedding: &[f32], templates: &[Vec<f32>]) -> f32 {
    match mode {
        TemplateScore::Max => templates
            .iter()
            .map(|t| cosine_similarity(embedding, t))
            .fold(0.0f32, f32::max),
        TemplateScore::Top3Mean => {
            let mut sims: Vec<f32> = templates
                .iter()
                .map(|t| cosine_similarity(embedding, t))
                .collect();
            sims.sort_by(|a, b| b.total_cmp(a));
            sims.truncate(TOP_K);
            if sims.is_empty() {
                0.0
            } else {
                sims.iter().sum::<f32>() / sims.len() as f32
            }
        }
        TemplateScore::Centroid => aggregate_embeddings(templates)
            .map(|c| cosine_similarity(embedding, &c))
            .unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vector at `deg` degrees in the first two axes.
    fn at(deg: f32) -> Vec<f32> {
        let r = deg.to_radians();
        vec![r.cos(), r.sin(), 0.0]
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn max_is_the_best_single_template() {
        let gallery = vec![at(0.0), at(60.0), at(120.0)];
        let probe = at(50.0);
        let s = template_score(TemplateScore::Max, &probe, &gallery);
        assert!(close(s, 10f32.to_radians().cos()), "{s}");
    }

    #[test]
    fn max_floors_at_zero_like_the_original_fold() {
        let s = template_score(TemplateScore::Max, &at(0.0), &[at(180.0)]);
        assert_eq!(s, 0.0);
        assert_eq!(template_score(TemplateScore::Max, &at(0.0), &[]), 0.0);
    }

    #[test]
    fn top3_mean_averages_the_three_best() {
        let gallery = vec![at(0.0), at(30.0), at(60.0), at(170.0)];
        let probe = at(0.0);
        let expected = (1.0 + 30f32.to_radians().cos() + 60f32.to_radians().cos()) / 3.0;
        assert!(close(
            template_score(TemplateScore::Top3Mean, &probe, &gallery),
            expected
        ));
        // Fewer than three templates: mean over what exists.
        let two = template_score(TemplateScore::Top3Mean, &probe, &[at(0.0), at(60.0)]);
        assert!(close(two, (1.0 + 60f32.to_radians().cos()) / 2.0));
        assert_eq!(template_score(TemplateScore::Top3Mean, &probe, &[]), 0.0);
    }

    #[test]
    fn centroid_scores_against_the_renormalised_mean() {
        let gallery = vec![at(-40.0), at(40.0)];
        // The centroid points at 0 degrees; a probe there scores 1.0, while
        // max would give cos(40 degrees).
        let probe = at(0.0);
        assert!(close(
            template_score(TemplateScore::Centroid, &probe, &gallery),
            1.0
        ));
        assert!(close(
            template_score(TemplateScore::Max, &probe, &gallery),
            40f32.to_radians().cos()
        ));
        assert_eq!(template_score(TemplateScore::Centroid, &probe, &[]), 0.0);
    }

    #[test]
    fn modes_order_for_a_spread_gallery() {
        // max >= top3_mean for any gallery (the best is at least the mean of the best three).
        let gallery = vec![at(10.0), at(70.0), at(100.0), at(150.0)];
        let probe = at(0.0);
        let max = template_score(TemplateScore::Max, &probe, &gallery);
        let top3 = template_score(TemplateScore::Top3Mean, &probe, &gallery);
        assert!(max >= top3);
    }
}
