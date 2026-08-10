use rand::Rng;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

/// Which way a critter faces, as an angle in radians measured clockwise from
/// north. An angle rather than one of eight compass points because turns are
/// as fine as fifteen degrees, and because a direction is what the rest of the
/// code wanted all along: the old integer offsets were a unit vector rounded
/// to whole pixels, and the diagonal correction was the error that rounding
/// introduced being taken back out again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heading {
    radians: f32,
}

pub const NORTH: Heading = Heading { radians: 0.0 };
pub const EAST: Heading = Heading { radians: FRAC_PI_2 };
pub const SOUTH: Heading = Heading { radians: PI };
pub const WEST: Heading = Heading {
    radians: 3.0 * FRAC_PI_2,
};
pub const NORTH_EAST: Heading = Heading {
    radians: FRAC_PI_2 / 2.0,
};
pub const SOUTH_EAST: Heading = Heading {
    radians: 3.0 * FRAC_PI_2 / 2.0,
};
pub const SOUTH_WEST: Heading = Heading {
    radians: 5.0 * FRAC_PI_2 / 2.0,
};
pub const NORTH_WEST: Heading = Heading {
    radians: 7.0 * FRAC_PI_2 / 2.0,
};

impl Heading {
    pub fn from_radians(radians: f32) -> Self {
        Self {
            radians: radians.rem_euclid(TAU),
        }
    }

    pub fn from_degrees(degrees: f32) -> Self {
        Self::from_radians(degrees.to_radians())
    }

    pub fn random<R: Rng>(rng: &mut R) -> Self {
        Self::from_radians(rng.gen_range(0.0..TAU))
    }

    pub fn radians(self) -> f32 {
        self.radians
    }

    /// Turned anticlockwise by `degrees`, which is what "left" means facing
    /// the way this heading points.
    pub fn turned_left(self, degrees: f32) -> Self {
        Self::from_radians(self.radians - degrees.to_radians())
    }

    pub fn turned_right(self, degrees: f32) -> Self {
        Self::from_radians(self.radians + degrees.to_radians())
    }

    /// A unit vector along this heading, with y increasing downward the way
    /// the screen does. Always one long, so nothing has to correct it.
    pub fn unit(self) -> (f32, f32) {
        (self.radians.sin(), -self.radians.cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Angles are compared loosely: they are built by rotating through
    // fractions of pi, so the last bits differ from the constants.
    fn assert_same(left: Heading, right: Heading) {
        let gap = (left.radians() - right.radians()).abs();
        let gap = gap.min(TAU - gap);
        assert!(
            gap < 1e-4,
            "expected {} to be {}",
            left.radians().to_degrees(),
            right.radians().to_degrees()
        );
    }

    mod turning {
        use super::*;

        #[test]
        fn turning_left_a_quarter_turn_from_north_faces_west() {
            assert_same(NORTH.turned_left(90.0), WEST);
        }

        #[test]
        fn turning_right_a_quarter_turn_from_north_faces_east() {
            assert_same(NORTH.turned_right(90.0), EAST);
        }

        #[test]
        fn turning_right_an_eighth_turn_from_north_faces_north_east() {
            assert_same(NORTH.turned_right(45.0), NORTH_EAST);
        }

        #[test]
        fn turning_left_past_north_wraps_around_the_compass() {
            assert_same(NORTH.turned_left(45.0), NORTH_WEST);
        }

        #[test]
        fn turning_right_past_west_wraps_around_the_compass() {
            assert_same(WEST.turned_right(135.0), NORTH_EAST);
        }

        #[test]
        fn turning_by_a_whole_turn_changes_nothing() {
            assert_same(NORTH_EAST.turned_right(360.0), NORTH_EAST);
        }

        #[test]
        fn a_turn_of_fifteen_degrees_is_a_turn_of_fifteen_degrees() {
            // The point of an angle rather than eight compass points: turns
            // finer than an eighth of a circle mean something.
            assert_same(NORTH.turned_right(15.0), Heading::from_degrees(15.0));
        }

        #[test]
        fn turning_left_and_then_right_by_the_same_amount_returns() {
            assert_same(NORTH.turned_left(15.0).turned_right(15.0), NORTH);
        }
    }

    mod unit_vector {
        use super::*;

        fn assert_close(actual: (f32, f32), expected: (f32, f32)) {
            assert!(
                (actual.0 - expected.0).abs() < 1e-4 && (actual.1 - expected.1).abs() < 1e-4,
                "expected {expected:?}, got {actual:?}"
            );
        }

        #[test]
        fn north_points_up_the_screen() {
            assert_close(NORTH.unit(), (0.0, -1.0));
        }

        #[test]
        fn east_points_right() {
            assert_close(EAST.unit(), (1.0, 0.0));
        }

        #[test]
        fn south_points_down_the_screen() {
            assert_close(SOUTH.unit(), (0.0, 1.0));
        }

        #[test]
        fn west_points_left() {
            assert_close(WEST.unit(), (-1.0, 0.0));
        }

        #[test]
        fn a_diagonal_is_still_one_long() {
            // What the old integer offsets could not manage: (1, -1) is a
            // step of root two, so every diagonal move needed correcting.
            let (dx, dy) = NORTH_EAST.unit();

            assert!((dx * dx + dy * dy - 1.0).abs() < 1e-4);
        }

        #[test]
        fn every_heading_is_one_long() {
            for degrees in (0..360).step_by(15) {
                let (dx, dy) = Heading::from_degrees(degrees as f32).unit();

                assert!(
                    (dx * dx + dy * dy - 1.0).abs() < 1e-4,
                    "{degrees} degrees was not a unit vector"
                );
            }
        }
    }

    mod construction {
        use super::*;
        use rand::SeedableRng;

        #[test]
        fn a_heading_reports_the_angle_it_was_built_from() {
            // Checked against a number rather than against another heading:
            // every other test here compares two headings, and a reader that
            // always returned the same thing would satisfy all of them.
            assert!((NORTH.radians() - 0.0).abs() < 1e-4);
            assert!((EAST.radians() - FRAC_PI_2).abs() < 1e-4);
            assert!((SOUTH.radians() - PI).abs() < 1e-4);
            assert!((Heading::from_degrees(15.0).radians() - 15f32.to_radians()).abs() < 1e-4);
        }

        #[test]
        fn an_angle_past_a_whole_turn_comes_back_round() {
            assert_same(Heading::from_degrees(360.0 + 45.0), NORTH_EAST);
        }

        #[test]
        fn a_negative_angle_comes_back_round_the_other_way() {
            assert_same(Heading::from_degrees(-45.0), NORTH_WEST);
        }

        #[test]
        fn a_random_heading_lies_within_one_turn() {
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);

            for _ in 0..100 {
                let heading = Heading::random(&mut rng);

                assert!((0.0..TAU).contains(&heading.radians()));
            }
        }
    }
}
