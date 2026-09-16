mod tests {
    use super::*;

    #[test]
    fn oklab_settings_reject_invalid_scales_and_preserve_serde_defaults() {
        let old: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(old.oklab_palette_threshold_scale, 1.0);
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, 10.1] {
            assert!(Config {
                oklab_palette_threshold_scale: scale,
                ..Config::default()
            }
            .validate()
            .is_err());
        }
        let config = Config {
            oklab_palette_threshold_scale: 0.5,
            ..Config::default()
        };
        let decoded: Config =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(decoded.oklab_palette_threshold_scale, 0.5);
        assert!(decoded.validate().is_ok());
    }

    #[test]
    fn tonal_budgets_tighten_smoothly_toward_both_extremes() {
        let config = Config::default();
        let knee = config.dark_knee_lightness;
        assert_eq!(config.tonal_detail_scale(knee), 1.0);
        assert!(config.tonal_detail_scale(0.0) < config.tonal_detail_scale(20.0));
        assert!(config.tonal_detail_scale(20.0) < config.tonal_detail_scale(knee));
        assert!(config.tonal_detail_scale(100.0) < config.tonal_detail_scale(80.0));
        assert!(config.tonal_detail_scale(80.0) < config.tonal_detail_scale(knee));
        for lightness in 0..100 {
            assert!(
                (config.tonal_detail_scale(lightness as f32 + 1.0)
                    - config.tonal_detail_scale(lightness as f32))
                .abs()
                    < 0.03
            );
        }
        for dark_knee_lightness in [0.0, 100.0] {
            let config = Config {
                dark_knee_lightness,
                ..Config::default()
            };
            for lightness in [0.0, 50.0, 100.0] {
                assert!((0.19..=1.0).contains(&config.tonal_detail_scale(lightness)));
            }
        }
    }

    #[test]
    fn default_configuration_is_valid() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn maximum_dimension_remains_a_hard_automatic_bound() {
        let config = Config {
            maximum_dimension: 320,
            auto_minimum_dimension: 768,
            auto_maximum_dimension: 1600,
            ..Config::default()
        };
        assert_eq!(config.automatic_dimension_bounds(), (320, 320));
    }

    #[test]
    fn nonfinite_threshold_is_rejected() {
        let config = Config {
            paint_primary_min_explained_variance: f32::NAN,
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            solid_color_max_delta_e: f32::INFINITY,
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            adaptive_min_predicted_rate: f32::NAN,
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }
}
