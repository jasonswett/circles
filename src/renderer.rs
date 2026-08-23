use crate::{Critter, Pellet, PELLET_RADIUS};

pub const ZERO_ENERGY_COLOR: u32 = 0x40_40_40;
pub const EATEN_COLOR: u32 = 0xFF_00_00;
pub const OUTLINE_THICKNESS: i32 = 2;
pub const FRONT_DOT_RADIUS: i32 = 4;

struct Ring {
    cx: i32,
    cy: i32,
    radius: i32,
    inner_squared: i32,
}

impl Ring {
    /// Whether any part of this ring's bounding box falls on the canvas.
    /// Cheap enough to be worth asking before walking the box row by row.
    fn touches(&self, canvas: &Canvas) -> bool {
        self.cx + self.radius >= 0
            && self.cy + self.radius >= 0
            && self.cx - self.radius < canvas.width as i32
            && self.cy - self.radius < canvas.height as i32
    }
}

struct Canvas<'a> {
    buffer: &'a mut [u32],
    width: usize,
    height: usize,
}

impl Canvas<'_> {
    /// Lights one pixel, wrapping at the edges the way the world does.
    fn set_with_wrap(&mut self, x: i32, y: i32, color: u32) {
        let x = x.rem_euclid(self.width as i32) as usize;
        let y = y.rem_euclid(self.height as i32) as usize;
        self.buffer[y * self.width + x] = color;
    }
}

pub struct Renderer;

impl Renderer {
    /// Draws a critter at the size its own energy gives it. The radius is not
    /// a parameter: feelers start at the body's edge and are placed from the
    /// critter's own radius, so a caller passing a different one would draw a
    /// body and feelers that disagreed about where the body ended.
    pub fn draw(critter: &Critter, buffer: &mut [u32], width: usize, height: usize) {
        let radius = critter.radius();
        let cx = critter.x();
        let cy = critter.y();
        let inner_radius = radius - OUTLINE_THICKNESS;
        let mut canvas = Canvas {
            buffer,
            width,
            height,
        };
        let color = if critter.is_being_eaten() {
            EATEN_COLOR
        } else {
            energy_color(
                critter.energy(),
                critter.initial_energy(),
                critter.genome_color(),
            )
        };

        let body = Ring {
            cx,
            cy,
            radius,
            inner_squared: inner_radius * inner_radius,
        };
        Self::fill_ring_with_wrap(&body, &mut canvas, color);

        let heading = critter.heading();
        let (offset_x, offset_y) = heading.unit();
        let dot_offset = (radius - FRONT_DOT_RADIUS) as f32;
        let dot = Ring {
            cx: cx + (offset_x * dot_offset).round() as i32,
            cy: cy + (offset_y * dot_offset).round() as i32,
            radius: FRONT_DOT_RADIUS,
            inner_squared: -1,
        };
        Self::fill_ring_with_wrap(&dot, &mut canvas, color);

        // A line out to each sensing disc, and the disc itself. The disc is
        // what actually feels anything, so drawing it at the size the genome
        // says makes a critter's reach readable rather than something to be
        // inferred from where its feelers point.
        let ((left_x, left_y), (right_x, right_y)) = critter.feeler_tips();
        let disc_radius = critter.feeler_disc().round() as i32;
        // Only the feelers the critter grew, so the picture says which ones a
        // lineage has climbed its way to.
        let grown = [
            (critter.has_left_feeler(), (left_x, left_y)),
            (critter.has_right_feeler(), (right_x, right_y)),
        ];
        for (tip_x, tip_y) in grown
            .into_iter()
            .filter_map(|(has, tip)| has.then_some(tip))
        {
            let (run, rise) = ((tip_x - cx) as f32, (tip_y - cy) as f32);
            let length = (run * run + rise * rise).sqrt();
            let steps = length.round().max(1.0) as i32;
            // Started at the body's edge rather than its middle, so the line
            // does not fill the hollow a critter's ring leaves.
            let first = ((radius as f32 / length) * steps as f32).round() as i32;
            for step in first..=steps {
                let along = step as f32 / steps as f32;
                canvas.set_with_wrap(
                    cx + (run * along).round() as i32,
                    cy + (rise * along).round() as i32,
                    color,
                );
            }
            let disc = Ring {
                cx: tip_x,
                cy: tip_y,
                radius: disc_radius,
                inner_squared: -1,
            };
            Self::fill_ring_with_wrap(&disc, &mut canvas, color);
        }
    }

    pub fn draw_pellet(pellet: &Pellet, buffer: &mut [u32], width: usize, height: usize) {
        let mut canvas = Canvas {
            buffer,
            width,
            height,
        };
        let disc = Ring {
            cx: pellet.x.round() as i32,
            cy: pellet.y.round() as i32,
            radius: PELLET_RADIUS,
            inner_squared: -1,
        };
        Self::fill_ring_with_wrap(&disc, &mut canvas, pellet.color());
    }

    // The `+` mutations on the offset additions below are equivalent: the loops
    // iterate over the symmetric set {-w, 0, w}, which is closed under negation,
    // so any sign flip produces the same set of draws in a different order.
    #[mutants::skip]
    fn fill_ring_with_wrap(ring: &Ring, canvas: &mut Canvas, color: u32) {
        let w = canvas.width as i32;
        let h = canvas.height as i32;
        for dx in [-w, 0, w] {
            for dy in [-h, 0, h] {
                let shifted = Ring {
                    cx: ring.cx + dx,
                    cy: ring.cy + dy,
                    radius: ring.radius,
                    inner_squared: ring.inner_squared,
                };
                // Of the nine copies only those overlapping the canvas can
                // show, and for a critter clear of the edges that is one. The
                // rest used to be walked row by row to draw nothing.
                if shifted.touches(canvas) {
                    Self::fill_ring(&shifted, canvas, color);
                }
            }
        }
    }

    /// Fills a row's worth of pixels at a time rather than asking of each one
    /// whether it lies inside. For a given row the shape spans a run either
    /// side of centre, and where that run's width is is arithmetic: a solid
    /// disc is one run, a hollow ring is two. Writing the runs turns a
    /// multiply, an add and a comparison per pixel into a square root per row.
    fn fill_ring(ring: &Ring, canvas: &mut Canvas, color: u32) {
        let outer_squared = ring.radius * ring.radius;
        let first_row = (ring.cy - ring.radius).max(0);
        let last_row = (ring.cy + ring.radius).min(canvas.height as i32 - 1);

        for y in first_row..=last_row {
            let dy = y - ring.cy;
            let outer_half = Self::half_width(outer_squared, dy);
            // The hollow's half-width where this row crosses it, so the run
            // either side of it can be written without testing the middle.
            let inner_half = Self::half_width(ring.inner_squared, dy);
            let row = y as usize * canvas.width;

            if inner_half.is_none() {
                // The row misses the hollow entirely: one unbroken run.
                if let Some(half) = outer_half {
                    Self::fill_span(canvas, row, ring.cx - half, ring.cx + half, color);
                }
                continue;
            }
            let (Some(outer), Some(inner)) = (outer_half, inner_half) else {
                continue;
            };
            Self::fill_span(canvas, row, ring.cx - outer, ring.cx - inner - 1, color);
            Self::fill_span(canvas, row, ring.cx + inner + 1, ring.cx + outer, color);
        }
    }

    /// How far either side of centre a circle of this squared radius reaches
    /// on the row `dy` from its middle, or None when the row misses it.
    fn half_width(radius_squared: i32, dy: i32) -> Option<i32> {
        let remaining = radius_squared - dy * dy;
        if remaining < 0 {
            return None;
        }
        Some((remaining as f64).sqrt() as i32)
    }

    /// Writes one unbroken run of a row, clipped to the canvas.
    fn fill_span(canvas: &mut Canvas, row: usize, from: i32, to: i32, color: u32) {
        let from = from.max(0) as usize;
        let to = to.min(canvas.width as i32 - 1);
        if to < 0 || from > to as usize {
            return;
        }
        canvas.buffer[row + from..=row + to as usize].fill(color);
    }
}

fn energy_color(energy: u32, initial_energy: u32, full_energy_color: u32) -> u32 {
    let ratio = if initial_energy == 0 {
        0.0
    } else {
        (energy as f32 / initial_energy as f32).clamp(0.0, 1.0)
    };
    interpolate_color(ZERO_ENERGY_COLOR, full_energy_color, ratio)
}

fn interpolate_color(from: u32, to: u32, ratio: f32) -> u32 {
    let [_, from_r, from_g, from_b] = from.to_be_bytes();
    let [_, to_r, to_g, to_b] = to.to_be_bytes();
    let r = lerp(from_r, to_r, ratio);
    let g = lerp(from_g, to_g, ratio);
    let b = lerp(from_b, to_b, ratio);
    u32::from_be_bytes([0, r, g, b])
}

fn lerp(from: u8, to: u8, ratio: f32) -> u8 {
    let from = from as f32;
    let to = to as f32;
    (from + (to - from) * ratio).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Critter, Genome, Heading, Instruction, EAST, MAX_FEELER_ANGLE, MAX_FEELER_DISC,
        MAX_FEELER_LENGTH, MIN_FEELER_DISC, MIN_FEELER_LENGTH, NORTH, NORTH_EAST, SOUTH,
        SOUTH_WEST, WEST,
    };

    mod interpolate_color_tests {
        use crate::renderer::interpolate_color;

        #[test]
        fn at_ratio_zero_it_returns_the_from_color() {
            assert_eq!(interpolate_color(0xAB_CD_EF, 0x12_34_56, 0.0), 0xAB_CD_EF);
        }

        #[test]
        fn at_ratio_one_it_returns_the_to_color() {
            assert_eq!(interpolate_color(0xAB_CD_EF, 0x12_34_56, 1.0), 0x12_34_56);
        }

        #[test]
        fn each_channel_is_interpolated_independently() {
            // from = (0, 100, 200), to = (200, 100, 0), ratio = 0.5.
            // r: 0 + (200 - 0) * 0.5 = 100
            // g: 100 + (100 - 100) * 0.5 = 100
            // b: 200 + (0 - 200) * 0.5 = 100
            assert_eq!(interpolate_color(0x0064C8, 0xC86400, 0.5), 0x646464);
        }
    }

    const RADIUS: i32 = 20;
    const CANVAS: usize = 200;
    const CENTER: i32 = 50;
    const NEAR_TOP: i32 = RADIUS;
    const NEAR_LEFT: i32 = RADIUS;
    const NEAR_BOTTOM: i32 = CANVAS as i32 - 1 - RADIUS;
    const NEAR_RIGHT: i32 = CANVAS as i32 - 1 - RADIUS;

    mod draw {
        use super::*;

        #[test]
        fn the_center_of_the_critter_is_not_filled() {
            let critter = stationary_critter(CENTER, CENTER, NORTH);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER, CENTER), 0);
        }

        #[test]
        fn a_point_well_inside_the_outline_is_not_filled() {
            // 5 pixels in from center is well inside the inner_radius of 18.
            let critter = stationary_critter(CENTER, CENTER, NORTH);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER + 5, CENTER + 5), 0);
        }

        #[test]
        fn the_pixel_just_inside_the_outer_radius_is_drawn_in_outline_color() {
            let critter = stationary_critter(CENTER, CENTER, NORTH);

            let buffer = render(&critter);

            // The outer edge: at distance radius - 1, distance² = (radius-1)² ≤ radius².
            assert_eq!(
                pixel_at(&buffer, CENTER + RADIUS - 1, CENTER),
                critter.genome_color()
            );
        }

        #[test]
        fn the_pixel_at_the_inner_radius_is_not_filled() {
            // The inner_radius is `radius - thickness`. A pixel exactly at distance
            // `inner_radius` has distance² == inner_squared, which fails the strict
            // `distance_squared > inner_squared` check, so it's outside the ring.
            let critter = stationary_critter(CENTER, CENTER, NORTH);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER + RADIUS - OUTLINE_THICKNESS, CENTER),
                0
            );
        }

        #[test]
        fn the_pixel_one_step_outside_the_inner_radius_is_drawn_in_outline_color() {
            let critter = stationary_critter(CENTER, CENTER, NORTH);

            let buffer = render(&critter);

            // One pixel further out than the inner_radius is the innermost lit pixel.
            assert_eq!(
                pixel_at(&buffer, CENTER + RADIUS - OUTLINE_THICKNESS + 1, CENTER),
                critter.genome_color()
            );
        }

        #[test]
        fn a_point_outside_the_outer_radius_is_not_drawn() {
            let critter = stationary_critter(CENTER, CENTER, NORTH);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER + RADIUS + 1, CENTER), 0);
        }

        #[test]
        fn the_front_dot_is_drawn_north_of_center_when_facing_north() {
            let critter = stationary_critter(CENTER, CENTER, NORTH);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER, CENTER - RADIUS + FRONT_DOT_RADIUS),
                critter.genome_color()
            );
        }

        #[test]
        fn the_front_dot_extends_to_its_full_radius_inside_the_ring() {
            // The pixel at the dot's bottom edge sits inside the ring's hollow center,
            // so it's only lit if the dot itself is at full radius.
            let critter = stationary_critter(CENTER, CENTER, NORTH);

            let buffer = render(&critter);

            let dot_center_y = CENTER - RADIUS + FRONT_DOT_RADIUS;
            assert_eq!(
                pixel_at(&buffer, CENTER, dot_center_y + FRONT_DOT_RADIUS),
                critter.genome_color()
            );
        }

        #[test]
        fn the_front_dot_is_drawn_east_of_center_when_facing_east() {
            let critter = stationary_critter(CENTER, CENTER, EAST);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER + RADIUS - FRONT_DOT_RADIUS, CENTER),
                critter.genome_color()
            );
        }

        #[test]
        fn the_front_dot_is_drawn_south_of_center_when_facing_south() {
            let critter = stationary_critter(CENTER, CENTER, SOUTH);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER, CENTER + RADIUS - FRONT_DOT_RADIUS),
                critter.genome_color()
            );
        }

        #[test]
        fn the_front_dot_is_drawn_west_of_center_when_facing_west() {
            let critter = stationary_critter(CENTER, CENTER, WEST);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER - RADIUS + FRONT_DOT_RADIUS, CENTER),
                critter.genome_color()
            );
        }

        #[test]
        fn the_front_dot_is_drawn_at_the_diagonal_offset_when_facing_northeast() {
            // For a diagonal heading, the dot's offset from center is scaled
            // by √2/2 along each axis. Derived rather than hard-coded: a dot
            // wide enough to cover its neighbours would let a literal keep
            // passing after the offset had moved.
            let diagonal_offset = (((RADIUS - FRONT_DOT_RADIUS) as f32)
                * std::f32::consts::FRAC_1_SQRT_2)
                .round() as i32;
            let critter = stationary_critter(CENTER, CENTER, NORTH_EAST);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER + diagonal_offset, CENTER - diagonal_offset),
                critter.genome_color()
            );
        }

        #[test]
        fn the_front_dot_is_drawn_at_the_diagonal_offset_when_facing_southwest() {
            let diagonal_offset = (((RADIUS - FRONT_DOT_RADIUS) as f32)
                * std::f32::consts::FRAC_1_SQRT_2)
                .round() as i32;
            let critter = stationary_critter(CENTER, CENTER, SOUTH_WEST);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER - diagonal_offset, CENTER + diagonal_offset),
                critter.genome_color()
            );
        }

        #[test]
        fn the_ring_extends_all_the_way_to_the_top_edge_when_the_critter_is_against_it() {
            let critter = stationary_critter(CENTER, NEAR_TOP, EAST);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER, 0), critter.genome_color());
        }

        #[test]
        fn the_ring_extends_all_the_way_to_the_left_edge_when_the_critter_is_against_it() {
            let critter = stationary_critter(NEAR_LEFT, CENTER, NORTH);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, 0, CENTER), critter.genome_color());
        }

        #[test]
        fn the_ring_extends_all_the_way_to_the_right_edge_when_the_critter_is_against_it() {
            let critter = stationary_critter(NEAR_RIGHT, CENTER, NORTH);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CANVAS as i32 - 1, CENTER),
                critter.genome_color()
            );
        }

        #[test]
        fn the_ring_extends_all_the_way_to_the_bottom_edge_when_the_critter_is_against_it() {
            let critter = stationary_critter(CENTER, NEAR_BOTTOM, NORTH);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER, CANVAS as i32 - 1),
                critter.genome_color()
            );
        }

        #[test]
        fn the_ring_is_still_drawn_when_part_of_the_critter_is_above_the_top_edge() {
            // Critter centered at y=0: top half of the ring is off-canvas, bottom half visible.
            let critter = stationary_critter(CENTER, 0, EAST);

            let buffer = render(&critter);

            // The bottom of the ring (at distance RADIUS below center) is on-canvas.
            assert_eq!(pixel_at(&buffer, CENTER, RADIUS), critter.genome_color());
        }

        #[test]
        fn the_ring_is_still_drawn_when_part_of_the_critter_is_left_of_the_left_edge() {
            let critter = stationary_critter(0, CENTER, NORTH);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, RADIUS, CENTER), critter.genome_color());
        }

        #[test]
        fn the_ring_is_still_drawn_when_part_of_the_critter_is_right_of_the_right_edge() {
            let critter = stationary_critter(CANVAS as i32 - 1, CENTER, NORTH);

            let buffer = render(&critter);

            // The left side of the ring (at distance RADIUS to the left) is on-canvas.
            assert_eq!(
                pixel_at(&buffer, CANVAS as i32 - 1 - RADIUS, CENTER),
                critter.genome_color()
            );
        }

        #[test]
        fn the_ring_is_still_drawn_when_part_of_the_critter_is_below_the_bottom_edge() {
            let critter = stationary_critter(CENTER, CANVAS as i32 - 1, NORTH);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER, CANVAS as i32 - 1 - RADIUS),
                critter.genome_color()
            );
        }

        mod color {
            use super::*;
            use crate::Instruction;

            const INITIAL_ENERGY: u32 = 100;
            // Derived from whatever radius the critter's own energy gives it,
            // since that is the size it is drawn at.
            fn on_ring_x_offset(critter: &Critter) -> i32 {
                critter.radius() - 1
            }

            #[test]
            fn a_critter_at_full_energy_renders_in_its_genome_color() {
                let critter = critter_with_energy(INITIAL_ENERGY);

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + on_ring_x_offset(&critter), CENTER),
                    critter.genome_color()
                );
            }

            #[test]
            fn a_critter_at_zero_energy_renders_in_ghostly_gray() {
                let critter = critter_with_energy(0);

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + on_ring_x_offset(&critter), CENTER),
                    0x40_40_40
                );
            }

            #[test]
            fn a_critter_at_half_energy_renders_halfway_between_gray_and_its_genome_color() {
                let critter = critter_with_energy(INITIAL_ENERGY / 2);

                let buffer = render(&critter);

                let halfway = halfway_between(0x40_40_40, critter.genome_color());
                assert_eq!(
                    pixel_at(&buffer, CENTER + on_ring_x_offset(&critter), CENTER),
                    halfway
                );
            }

            #[test]
            fn a_critter_with_energy_above_initial_still_renders_in_its_genome_color() {
                // With the cap on gain_energy removed, a critter can stockpile
                // past its initial_energy. The color must clamp at the full
                // genome color rather than continuing to brighten.
                let mut critter = critter_with_energy(INITIAL_ENERGY);
                critter.gain_energy(INITIAL_ENERGY); // total = 2 × INITIAL_ENERGY

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + on_ring_x_offset(&critter), CENTER),
                    critter.genome_color()
                );
            }

            fn halfway_between(from: u32, to: u32) -> u32 {
                // Match the renderer's lerp, which rounds to nearest rather
                // than truncating, so odd-sum channels don't diverge.
                let [_, fr, fg, fb] = from.to_be_bytes();
                let [_, tr, tg, tb] = to.to_be_bytes();
                let lerp_half =
                    |a: u8, b: u8| -> u8 { (a as f32 + (b as f32 - a as f32) * 0.5).round() as u8 };
                u32::from_be_bytes([0, lerp_half(fr, tr), lerp_half(fg, tg), lerp_half(fb, tb)])
            }

            #[test]
            fn a_critter_being_eaten_renders_in_red_regardless_of_energy() {
                let mut critter = critter_with_energy(INITIAL_ENERGY);
                critter.mark_being_eaten_for(10);

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + on_ring_x_offset(&critter), CENTER),
                    0xFF_00_00
                );
            }

            fn critter_with_energy(current_energy: u32) -> Critter {
                // Drained in one step rather than by ticking it down: upkeep
                // takes a share of what a critter holds, so ticking away a
                // fixed amount would overshoot and never land where asked.
                let mut critter = Critter::with_genome(
                    CENTER,
                    CENTER,
                    NORTH,
                    1,
                    1,
                    INITIAL_ENERGY,
                    0,
                    Genome::all(Instruction::DoNothing),
                );
                critter.lose_energy(INITIAL_ENERGY - current_energy);
                critter
            }
        }

        mod feelers {
            use super::*;

            // An ordinary energy rather than u32::MAX: size grows with
            // energy, and a critter that rich is thousands of pixels across,
            // so its feelers land nowhere near the test canvas.
            const FED: u32 = 2_500;

            fn feeler_critter(length: f32, angle: f32, disc: f32) -> Critter {
                let mut genome = Genome::all(Instruction::DoNothing);
                genome.set_feeler_shape(length, angle, disc);
                genome.set_feeler_count(2);
                Critter::with_genome(CENTER, CENTER, NORTH, 1, 1, FED, 0, genome)
            }

            #[test]
            fn a_feeler_a_critter_never_grew_is_not_drawn() {
                // What is drawn is what the critter has, so the picture says
                // which feelers a lineage has climbed its way to.
                let mut genome = Genome::all(Instruction::DoNothing);
                genome.set_feeler_shape(20.0, MAX_FEELER_ANGLE, 6.0);
                genome.set_feeler_count(1);
                let critter = Critter::with_genome(CENTER, CENTER, NORTH, 1, 1, FED, 0, genome);
                let ((lx, ly), _) = critter.feeler_tips();
                // Where the missing feeler would have been had the critter
                // grown it: a lone feeler points straight ahead, so the two
                // tips coincide and only the splayed pair tells them apart.
                let mut paired = Genome::all(Instruction::DoNothing);
                paired.set_feeler_shape(20.0, MAX_FEELER_ANGLE, 6.0);
                paired.set_feeler_count(2);
                let with_both = Critter::with_genome(CENTER, CENTER, NORTH, 1, 1, FED, 0, paired);
                let (_, (rx, ry)) = with_both.feeler_tips();

                let buffer = render(&critter);

                assert_eq!(pixel_at(&buffer, lx, ly), critter.genome_color());
                assert_eq!(pixel_at(&buffer, rx, ry), 0);
            }

            #[test]
            fn a_critter_draws_a_disc_at_each_feeler_tip() {
                // The disc is what senses, so the disc is what has to be
                // visible: a critter's reach should be readable off the
                // screen rather than inferred.
                let critter = feeler_critter(20.0, 45.0, 6.0);
                let ((lx, ly), (rx, ry)) = critter.feeler_tips();

                let buffer = render(&critter);

                assert_eq!(pixel_at(&buffer, lx, ly), critter.genome_color());
                assert_eq!(pixel_at(&buffer, rx, ry), critter.genome_color());
            }

            #[test]
            fn a_line_runs_out_to_each_disc() {
                let critter = feeler_critter(40.0, 45.0, 4.0);
                let ((lx, ly), _) = critter.feeler_tips();

                let buffer = render(&critter);

                // Partway along, beyond the body and short of the disc.
                let midx = (CENTER + lx) / 2;
                let midy = (CENTER + ly) / 2;
                assert_eq!(pixel_at(&buffer, midx, midy), critter.genome_color());
            }

            #[test]
            fn the_line_starts_at_the_body_and_not_before() {
                // The body is a ring, so a line drawn from the centre would
                // fill the hollow it leaves. Pins where the line begins:
                // one step short of the body's edge must still be dark.
                // Feelers held out to the side, so the head dot -- which sits
                // ahead of the critter and is nearly as wide as a small body
                // -- cannot be what lights the pixels being checked.
                let critter = feeler_critter(30.0, MAX_FEELER_ANGLE, MIN_FEELER_DISC);

                let buffer = render(&critter);

                let radius = critter.radius();
                // A right angle to the left of north is due west. Sampled
                // just inside the ring's inner edge, which is the first pixel
                // of the hollow a line starting early would fill.
                assert_eq!(
                    pixel_at(&buffer, CENTER - radius + OUTLINE_THICKNESS + 1, CENTER),
                    0,
                    "the line should not begin inside the body"
                );
                assert_eq!(
                    pixel_at(&buffer, CENTER - radius, CENTER),
                    critter.genome_color(),
                    "the line should start at the body's edge"
                );
            }

            #[test]
            fn the_line_runs_the_whole_way_to_the_disc() {
                // And ends where the disc begins, so there is no gap between
                // a critter and what it is feeling with.
                let critter = feeler_critter(30.0, 0.0, MIN_FEELER_DISC);
                let ((_, ly), _) = critter.feeler_tips();

                let buffer = render(&critter);

                // Every pixel from the body's edge to the tip is lit.
                for y in ly..=(CENTER - critter.radius()) {
                    assert_eq!(
                        pixel_at(&buffer, CENTER, y),
                        critter.genome_color(),
                        "gap in the feeler at y={y}"
                    );
                }
            }

            #[test]
            fn a_longer_feeler_draws_a_longer_line() {
                // The line's length follows the genome rather than being
                // fixed: what is drawn has to say what the critter can reach.
                let short = feeler_critter(MIN_FEELER_LENGTH, 0.0, MIN_FEELER_DISC);
                let long = feeler_critter(MAX_FEELER_LENGTH, 0.0, MIN_FEELER_DISC);

                let short_buffer = render(&short);
                let long_buffer = render(&long);

                let lit = |buffer: &[u32], critter: &Critter| {
                    (1..CENTER)
                        .filter(|&y| pixel_at(buffer, CENTER, CENTER - y) == critter.genome_color())
                        .count()
                };
                assert!(
                    lit(&long_buffer, &long) > lit(&short_buffer, &short),
                    "the longer feeler should light more pixels"
                );
            }

            #[test]
            fn the_discs_are_solid_rather_than_hollow() {
                // A disc, not a ring: what a feeler senses is the whole patch,
                // so the whole patch is filled. Sampled off-centre, since the
                // very middle is lit either way.
                let critter = feeler_critter(30.0, 45.0, MAX_FEELER_DISC);
                let ((lx, ly), _) = critter.feeler_tips();

                let buffer = render(&critter);

                for offset in 1..=(MAX_FEELER_DISC as i32 - 2) {
                    assert_eq!(
                        pixel_at(&buffer, lx + offset, ly),
                        critter.genome_color(),
                        "the disc should be filled at {offset} from its middle"
                    );
                }
            }

            #[test]
            fn each_feeler_is_drawn_on_its_own_side() {
                // Held at an angle that puts the two tips at different
                // distances from the critter on each axis, so a line drawn
                // towards the wrong side lands somewhere nothing is drawn.
                // Facing a diagonal with the feelers held wide, so the two
                // lines are not mirror images of each other: with a critter
                // facing north and its feelers at equal angles, mirroring one
                // line lands it exactly on the other and nothing can tell.
                let mut genome = Genome::all(Instruction::DoNothing);
                genome.set_feeler_shape(30.0, MAX_FEELER_ANGLE, MIN_FEELER_DISC);
                genome.set_feeler_count(2);
                let critter =
                    Critter::with_genome(CENTER, CENTER, NORTH_EAST, 1, 1, FED, 0, genome);
                let ((lx, ly), (rx, ry)) = critter.feeler_tips();

                let buffer = render(&critter);

                // The tips straddle the critter, so a mirrored line would put
                // the left feeler's pixels where the right one's belong.
                assert!(lx < CENTER && ry > CENTER, "tips should sit apart");
                assert_eq!(pixel_at(&buffer, lx, ly), critter.genome_color());
                assert_eq!(pixel_at(&buffer, rx, ry), critter.genome_color());
                // Partway along the left feeler's line, which is what a
                // mirrored line would move: the tips themselves are drawn by
                // the discs and would still land correctly.
                let midx = (CENTER + lx) / 2;
                let midy = (CENTER + ly) / 2;
                assert_eq!(
                    pixel_at(&buffer, midx, midy),
                    critter.genome_color(),
                    "the left feeler's line should run towards its own tip"
                );
                // Mirroring the left feeler about the critter's x lands on the
                // right feeler, which is drawn there legitimately. What tells
                // them apart is that the left line's own pixels stop being
                // lit if it is mirrored, so this counts them.
                let left_pixels = (1..40)
                    .filter(|&step| {
                        let along = step as f32 / 40.0;
                        let x = CENTER + ((lx - CENTER) as f32 * along).round() as i32;
                        let y = CENTER + ((ly - CENTER) as f32 * along).round() as i32;
                        pixel_at(&buffer, x, y) == critter.genome_color()
                    })
                    .count();
                assert!(
                    left_pixels > 20,
                    "most of the left feeler's line should be lit, {left_pixels} were"
                );
            }

            #[test]
            fn the_discs_are_drawn_the_size_the_genome_says() {
                // A bigger disc has to look bigger, or the picture stops
                // saying what the critter can feel.
                let small = feeler_critter(20.0, 45.0, MIN_FEELER_DISC);
                let large = feeler_critter(20.0, 45.0, MAX_FEELER_DISC);
                let ((lx, ly), _) = large.feeler_tips();
                // Just inside the large disc, outside the small one.
                let probe = MAX_FEELER_DISC as i32 - 1;

                let small_buffer = render(&small);
                let large_buffer = render(&large);

                assert_eq!(
                    pixel_at(&large_buffer, lx + probe, ly),
                    large.genome_color()
                );
                let ((sx, sy), _) = small.feeler_tips();
                assert_eq!(pixel_at(&small_buffer, sx + probe, sy), 0);
            }

            #[test]
            fn the_feelers_are_drawn_in_the_critters_own_colour() {
                // Part of the animal, so they dim along with it.
                let mut critter = feeler_critter(20.0, 45.0, 5.0);
                critter.lose_energy(critter.energy() / 2);
                let ((lx, ly), _) = critter.feeler_tips();

                let buffer = render(&critter);

                let body = pixel_at(&buffer, CENTER + critter.radius() - 1, CENTER);
                assert_eq!(pixel_at(&buffer, lx, ly), body);
            }

            #[test]
            fn the_feelers_turn_with_the_critter() {
                let mut genome = Genome::all(Instruction::DoNothing);
                genome.set_feeler_shape(20.0, MAX_FEELER_ANGLE, 5.0);
                genome.set_feeler_count(2);
                let critter = Critter::with_genome(CENTER, CENTER, EAST, 1, 1, FED, 0, genome);
                let ((lx, ly), (rx, ry)) = critter.feeler_tips();

                let buffer = render(&critter);

                // Facing east, a right angle either side points north and south.
                assert!(
                    ly < CENTER && ry > CENTER,
                    "tips should straddle the critter"
                );
                assert_eq!(pixel_at(&buffer, lx, ly), critter.genome_color());
                assert_eq!(pixel_at(&buffer, rx, ry), critter.genome_color());
            }
        }

        mod fill_ring {
            use super::*;

            // The shape fill_ring is meant to draw, worked out the slow and
            // obvious way: every pixel of the bounding box, tested against the
            // two radii. Whatever fill_ring does to go faster, it has to agree
            // with this exactly.
            fn drawn_the_obvious_way(ring: &Ring, color: u32) -> Vec<u32> {
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                let outer = ring.radius * ring.radius;
                for y in (ring.cy - ring.radius)..=(ring.cy + ring.radius) {
                    for x in (ring.cx - ring.radius)..=(ring.cx + ring.radius) {
                        if x < 0 || y < 0 || x >= CANVAS as i32 || y >= CANVAS as i32 {
                            continue;
                        }
                        let (dx, dy) = (x - ring.cx, y - ring.cy);
                        let distance = dx * dx + dy * dy;
                        if distance <= outer && distance > ring.inner_squared {
                            buffer[y as usize * CANVAS + x as usize] = color;
                        }
                    }
                }
                buffer
            }

            fn drawn_by_fill_ring(ring: &Ring, color: u32) -> Vec<u32> {
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                let mut canvas = Canvas {
                    buffer: &mut buffer,
                    width: CANVAS,
                    height: CANVAS,
                };
                Renderer::fill_ring(ring, &mut canvas, color);
                buffer
            }

            fn assert_same_shape(ring: &Ring) {
                let color = 0x00_AB_CD_EF;
                assert_eq!(
                    drawn_by_fill_ring(ring, color),
                    drawn_the_obvious_way(ring, color),
                    "ring at ({}, {}) radius {} inner {}",
                    ring.cx,
                    ring.cy,
                    ring.radius,
                    ring.inner_squared
                );
            }

            #[test]
            fn a_solid_disc_is_drawn_pixel_for_pixel() {
                assert_same_shape(&Ring {
                    cx: CENTER,
                    cy: CENTER,
                    radius: 8,
                    inner_squared: -1,
                });
            }

            #[test]
            fn a_hollow_ring_is_drawn_pixel_for_pixel() {
                assert_same_shape(&Ring {
                    cx: CENTER,
                    cy: CENTER,
                    radius: 20,
                    inner_squared: 18 * 18,
                });
            }

            #[test]
            fn every_radius_is_drawn_pixel_for_pixel() {
                // Small radii are where a span worked out with square roots is
                // most likely to disagree with a pixel-by-pixel test.
                for radius in 0..=24 {
                    assert_same_shape(&Ring {
                        cx: CENTER,
                        cy: CENTER,
                        radius,
                        inner_squared: -1,
                    });
                }
            }

            #[test]
            fn every_thickness_of_ring_is_drawn_pixel_for_pixel() {
                for inner in 0..20 {
                    assert_same_shape(&Ring {
                        cx: CENTER,
                        cy: CENTER,
                        radius: 20,
                        inner_squared: inner * inner,
                    });
                }
            }

            #[test]
            fn a_ring_hanging_off_each_edge_is_drawn_pixel_for_pixel() {
                // Clipping is where a span-based fill is easiest to get wrong,
                // since the row it would write runs past the buffer.
                for (cx, cy) in [
                    (0, CENTER),
                    (CANVAS as i32 - 1, CENTER),
                    (CENTER, 0),
                    (CENTER, CANVAS as i32 - 1),
                    (0, 0),
                    (CANVAS as i32 - 1, CANVAS as i32 - 1),
                    (-5, CENTER),
                    (CANVAS as i32 + 5, CENTER),
                ] {
                    assert_same_shape(&Ring {
                        cx,
                        cy,
                        radius: 12,
                        inner_squared: -1,
                    });
                }
            }

            #[test]
            fn a_ring_entirely_off_the_canvas_draws_nothing() {
                let ring = Ring {
                    cx: -100,
                    cy: -100,
                    radius: 12,
                    inner_squared: -1,
                };

                assert!(drawn_by_fill_ring(&ring, 0x00_FF_FF_FF)
                    .iter()
                    .all(|&p| p == 0));
            }
        }

        mod offscreen_copies {
            use super::*;

            fn canvas_of(buffer: &mut Vec<u32>) -> Canvas<'_> {
                Canvas {
                    buffer,
                    width: CANVAS,
                    height: CANVAS,
                }
            }

            fn ring_at(cx: i32, cy: i32) -> Ring {
                Ring {
                    cx,
                    cy,
                    radius: RADIUS,
                    inner_squared: -1,
                }
            }

            #[test]
            fn a_ring_on_the_canvas_is_drawn() {
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                let canvas = canvas_of(&mut buffer);

                assert!(ring_at(CENTER, CENTER).touches(&canvas));
            }

            #[test]
            fn a_ring_a_whole_canvas_away_is_not() {
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                let canvas = canvas_of(&mut buffer);

                assert!(!ring_at(CENTER - CANVAS as i32, CENTER).touches(&canvas));
                assert!(!ring_at(CENTER + CANVAS as i32, CENTER).touches(&canvas));
                assert!(!ring_at(CENTER, CENTER - CANVAS as i32).touches(&canvas));
                assert!(!ring_at(CENTER, CENTER + CANVAS as i32).touches(&canvas));
            }

            #[test]
            fn a_ring_reaching_the_canvas_by_one_pixel_is_drawn() {
                // The boundary: its edge lands on the first column, so it has
                // something to show and must not be skipped.
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                let canvas = canvas_of(&mut buffer);

                assert!(ring_at(-RADIUS, CENTER).touches(&canvas));
                assert!(ring_at(CANVAS as i32 + RADIUS - 1, CENTER).touches(&canvas));
            }

            #[test]
            fn a_ring_one_pixel_short_of_the_canvas_is_not() {
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                let canvas = canvas_of(&mut buffer);

                assert!(!ring_at(-RADIUS - 1, CENTER).touches(&canvas));
                assert!(!ring_at(CANVAS as i32 + RADIUS, CENTER).touches(&canvas));
            }

            #[test]
            fn the_vertical_edges_are_judged_the_same_way() {
                // The same boundary along y. Tested separately because the two
                // axes are separate comparisons, and one can be got wrong
                // while the other is right.
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                let canvas = canvas_of(&mut buffer);

                assert!(ring_at(CENTER, -RADIUS).touches(&canvas));
                assert!(!ring_at(CENTER, -RADIUS - 1).touches(&canvas));
                assert!(ring_at(CENTER, CANVAS as i32 + RADIUS - 1).touches(&canvas));
                assert!(!ring_at(CENTER, CANVAS as i32 + RADIUS).touches(&canvas));
            }
        }

        mod wrap_rendering {
            use super::*;

            #[test]
            fn a_critter_against_the_right_edge_renders_pixels_on_the_left_edge_too() {
                // Critter at x = CANVAS - 1: ring extends from x = CANVAS - 1 - RADIUS to
                // x = CANVAS - 1 + RADIUS, the latter wrapping into [0, RADIUS - 1].
                let critter = stationary_critter(CANVAS as i32 - 1, CENTER, NORTH);

                let buffer = render(&critter);

                // A pixel that would be lit on the unwrapped circle's right side now
                // appears on the left edge.
                assert_eq!(
                    pixel_at(&buffer, RADIUS - 2, CENTER),
                    critter.genome_color()
                );
            }

            #[test]
            fn a_critter_straddling_a_corner_lights_all_four_corners() {
                // The case the skip must not break: at a corner four copies
                // are genuinely on the canvas at once, so a test that only
                // proved the far ones were dropped would be the wrong test.
                let critter = stationary_critter(0, 0, NORTH);

                let buffer = render(&critter);

                // Sampled on the ring itself rather than at the corner pixel,
                // which sits in the hollow centre of a body that is a ring and
                // not a disc.
                let on_ring = RADIUS - 1;
                let wrapped = CANVAS as i32 - on_ring;
                assert_eq!(pixel_at(&buffer, on_ring, 0), critter.genome_color());
                assert_eq!(pixel_at(&buffer, wrapped, 0), critter.genome_color());
                assert_eq!(pixel_at(&buffer, 0, on_ring), critter.genome_color());
                assert_eq!(pixel_at(&buffer, 0, wrapped), critter.genome_color());
            }

            #[test]
            fn a_critter_against_the_bottom_edge_renders_pixels_on_the_top_edge_too() {
                let critter = stationary_critter(CENTER, CANVAS as i32 - 1, NORTH);

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER, RADIUS - 2),
                    critter.genome_color()
                );
            }
        }

        mod pellet {
            use super::*;
            use crate::{Pellet, PELLET_COLOR, PELLET_RADIUS};

            #[test]
            fn a_poison_pellet_is_drawn_in_poison_color() {
                let pellet = Pellet::poison_at(CENTER, CENTER);
                let buffer = render_pellet(&pellet);

                assert_eq!(pixel_at(&buffer, CENTER, CENTER), crate::POISON_COLOR);
            }

            #[test]
            fn the_center_of_a_pellet_is_drawn_in_pellet_color() {
                let pellet = Pellet::at(CENTER, CENTER);
                let buffer = render_pellet(&pellet);

                assert_eq!(pixel_at(&buffer, CENTER, CENTER), PELLET_COLOR);
            }

            #[test]
            fn the_pellet_is_drawn_as_a_filled_disc_of_pellet_radius() {
                let pellet = Pellet::at(CENTER, CENTER);
                let buffer = render_pellet(&pellet);

                // Edge of the disc — at distance PELLET_RADIUS from center, on-axis.
                assert_eq!(
                    pixel_at(&buffer, CENTER + PELLET_RADIUS, CENTER),
                    PELLET_COLOR
                );
                // One past the edge — should not be drawn.
                assert_eq!(pixel_at(&buffer, CENTER + PELLET_RADIUS + 1, CENTER), 0);
            }

            #[test]
            fn drawing_a_pellet_does_not_panic_when_partly_off_canvas() {
                let pellet = Pellet::at(0, 0);
                let _ = render_pellet(&pellet);
            }

            fn render_pellet(pellet: &Pellet) -> Vec<u32> {
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                Renderer::draw_pellet(pellet, &mut buffer, CANVAS, CANVAS);
                buffer
            }
        }

        // Helpers below the tests, in keeping with hiding incidental detail.

        // Energy chosen so the critter's own radius comes out at RADIUS,
        // since that is what it is drawn at: size follows energy, so a test
        // critter cannot pick the two independently.
        const ENERGY_FOR_TEST_RADIUS: u32 = 25_280;

        fn stationary_critter(x: i32, y: i32, heading: Heading) -> Critter {
            let critter = Critter::with_genome(
                x,
                y,
                heading,
                1,
                1,
                ENERGY_FOR_TEST_RADIUS,
                0,
                Genome::all(Instruction::DoNothing),
            );
            debug_assert_eq!(critter.radius(), RADIUS, "energy no longer gives RADIUS");
            critter
        }

        fn render(critter: &Critter) -> Vec<u32> {
            let mut buffer = vec![0u32; CANVAS * CANVAS];
            Renderer::draw(critter, &mut buffer, CANVAS, CANVAS);
            buffer
        }

        fn pixel_at(buffer: &[u32], x: i32, y: i32) -> u32 {
            buffer[y as usize * CANVAS + x as usize]
        }
    }
}
