#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heading {
    North,
    East,
    South,
    West,
}

impl Heading {
    pub fn turn_left(self) -> Self {
        match self {
            Heading::North => Heading::West,
            Heading::West => Heading::South,
            Heading::South => Heading::East,
            Heading::East => Heading::North,
        }
    }

    pub fn turn_right(self) -> Self {
        match self {
            Heading::North => Heading::East,
            Heading::East => Heading::South,
            Heading::South => Heading::West,
            Heading::West => Heading::North,
        }
    }

    pub fn offset(self) -> (i32, i32) {
        match self {
            Heading::North => (0, -1),
            Heading::East => (1, 0),
            Heading::South => (0, 1),
            Heading::West => (-1, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod turn_left {
        use super::*;

        #[test]
        fn north_becomes_west() {
            assert_eq!(Heading::North.turn_left(), Heading::West);
        }

        #[test]
        fn west_becomes_south() {
            assert_eq!(Heading::West.turn_left(), Heading::South);
        }

        #[test]
        fn south_becomes_east() {
            assert_eq!(Heading::South.turn_left(), Heading::East);
        }

        #[test]
        fn east_becomes_north() {
            assert_eq!(Heading::East.turn_left(), Heading::North);
        }
    }

    mod turn_right {
        use super::*;

        #[test]
        fn north_becomes_east() {
            assert_eq!(Heading::North.turn_right(), Heading::East);
        }

        #[test]
        fn east_becomes_south() {
            assert_eq!(Heading::East.turn_right(), Heading::South);
        }

        #[test]
        fn south_becomes_west() {
            assert_eq!(Heading::South.turn_right(), Heading::West);
        }

        #[test]
        fn west_becomes_north() {
            assert_eq!(Heading::West.turn_right(), Heading::North);
        }
    }

    mod offset {
        use super::*;

        #[test]
        fn north_points_up() {
            assert_eq!(Heading::North.offset(), (0, -1));
        }

        #[test]
        fn east_points_right() {
            assert_eq!(Heading::East.offset(), (1, 0));
        }

        #[test]
        fn south_points_down() {
            assert_eq!(Heading::South.offset(), (0, 1));
        }

        #[test]
        fn west_points_left() {
            assert_eq!(Heading::West.offset(), (-1, 0));
        }
    }
}
