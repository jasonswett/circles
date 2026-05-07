use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Heading {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

impl Heading {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        match rng.gen_range(0..8) {
            0 => Heading::North,
            1 => Heading::NorthEast,
            2 => Heading::East,
            3 => Heading::SouthEast,
            4 => Heading::South,
            5 => Heading::SouthWest,
            6 => Heading::West,
            _ => Heading::NorthWest,
        }
    }

    pub fn turn_left(self) -> Self {
        match self {
            Heading::North => Heading::NorthWest,
            Heading::NorthWest => Heading::West,
            Heading::West => Heading::SouthWest,
            Heading::SouthWest => Heading::South,
            Heading::South => Heading::SouthEast,
            Heading::SouthEast => Heading::East,
            Heading::East => Heading::NorthEast,
            Heading::NorthEast => Heading::North,
        }
    }

    pub fn turn_right(self) -> Self {
        match self {
            Heading::North => Heading::NorthEast,
            Heading::NorthEast => Heading::East,
            Heading::East => Heading::SouthEast,
            Heading::SouthEast => Heading::South,
            Heading::South => Heading::SouthWest,
            Heading::SouthWest => Heading::West,
            Heading::West => Heading::NorthWest,
            Heading::NorthWest => Heading::North,
        }
    }

    pub fn offset(self) -> (i32, i32) {
        match self {
            Heading::North => (0, -1),
            Heading::NorthEast => (1, -1),
            Heading::East => (1, 0),
            Heading::SouthEast => (1, 1),
            Heading::South => (0, 1),
            Heading::SouthWest => (-1, 1),
            Heading::West => (-1, 0),
            Heading::NorthWest => (-1, -1),
        }
    }

    pub fn is_diagonal(self) -> bool {
        let (dx, dy) = self.offset();
        dx != 0 && dy != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod turn_left {
        use super::*;

        #[test]
        fn north_becomes_northwest() {
            assert_eq!(Heading::North.turn_left(), Heading::NorthWest);
        }

        #[test]
        fn northwest_becomes_west() {
            assert_eq!(Heading::NorthWest.turn_left(), Heading::West);
        }

        #[test]
        fn west_becomes_southwest() {
            assert_eq!(Heading::West.turn_left(), Heading::SouthWest);
        }

        #[test]
        fn southwest_becomes_south() {
            assert_eq!(Heading::SouthWest.turn_left(), Heading::South);
        }

        #[test]
        fn south_becomes_southeast() {
            assert_eq!(Heading::South.turn_left(), Heading::SouthEast);
        }

        #[test]
        fn southeast_becomes_east() {
            assert_eq!(Heading::SouthEast.turn_left(), Heading::East);
        }

        #[test]
        fn east_becomes_northeast() {
            assert_eq!(Heading::East.turn_left(), Heading::NorthEast);
        }

        #[test]
        fn northeast_becomes_north() {
            assert_eq!(Heading::NorthEast.turn_left(), Heading::North);
        }
    }

    mod turn_right {
        use super::*;

        #[test]
        fn north_becomes_northeast() {
            assert_eq!(Heading::North.turn_right(), Heading::NorthEast);
        }

        #[test]
        fn northeast_becomes_east() {
            assert_eq!(Heading::NorthEast.turn_right(), Heading::East);
        }

        #[test]
        fn east_becomes_southeast() {
            assert_eq!(Heading::East.turn_right(), Heading::SouthEast);
        }

        #[test]
        fn southeast_becomes_south() {
            assert_eq!(Heading::SouthEast.turn_right(), Heading::South);
        }

        #[test]
        fn south_becomes_southwest() {
            assert_eq!(Heading::South.turn_right(), Heading::SouthWest);
        }

        #[test]
        fn southwest_becomes_west() {
            assert_eq!(Heading::SouthWest.turn_right(), Heading::West);
        }

        #[test]
        fn west_becomes_northwest() {
            assert_eq!(Heading::West.turn_right(), Heading::NorthWest);
        }

        #[test]
        fn northwest_becomes_north() {
            assert_eq!(Heading::NorthWest.turn_right(), Heading::North);
        }
    }

    mod offset {
        use super::*;

        #[test]
        fn north_points_up() {
            assert_eq!(Heading::North.offset(), (0, -1));
        }

        #[test]
        fn northeast_points_up_and_right() {
            assert_eq!(Heading::NorthEast.offset(), (1, -1));
        }

        #[test]
        fn east_points_right() {
            assert_eq!(Heading::East.offset(), (1, 0));
        }

        #[test]
        fn southeast_points_down_and_right() {
            assert_eq!(Heading::SouthEast.offset(), (1, 1));
        }

        #[test]
        fn south_points_down() {
            assert_eq!(Heading::South.offset(), (0, 1));
        }

        #[test]
        fn southwest_points_down_and_left() {
            assert_eq!(Heading::SouthWest.offset(), (-1, 1));
        }

        #[test]
        fn west_points_left() {
            assert_eq!(Heading::West.offset(), (-1, 0));
        }

        #[test]
        fn northwest_points_up_and_left() {
            assert_eq!(Heading::NorthWest.offset(), (-1, -1));
        }
    }

    mod is_diagonal {
        use super::*;

        #[test]
        fn cardinal_directions_are_not_diagonal() {
            assert!(!Heading::North.is_diagonal());
            assert!(!Heading::East.is_diagonal());
            assert!(!Heading::South.is_diagonal());
            assert!(!Heading::West.is_diagonal());
        }

        #[test]
        fn ordinal_directions_are_diagonal() {
            assert!(Heading::NorthEast.is_diagonal());
            assert!(Heading::SouthEast.is_diagonal());
            assert!(Heading::SouthWest.is_diagonal());
            assert!(Heading::NorthWest.is_diagonal());
        }
    }

    mod random {
        use super::*;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[test]
        fn over_many_draws_every_heading_appears() {
            let mut rng = StdRng::seed_from_u64(42);
            let mut seen = std::collections::HashSet::new();

            for _ in 0..1000 {
                seen.insert(Heading::random(&mut rng));
            }

            assert_eq!(seen.len(), 8);
        }
    }
}
