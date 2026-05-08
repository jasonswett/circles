use circles::{Renderer, World, CRITTER_RADIUS};
use rand::rngs::StdRng;
use rand::SeedableRng;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
const FRAMES: usize = 200;

#[test]
fn the_simulation_can_run_for_many_frames_without_panicking() {
    let mut rng = StdRng::seed_from_u64(0);
    let mut world = World::new(WIDTH, HEIGHT, &mut rng);
    let mut frame_pixels = vec![0u32; WIDTH * HEIGHT];

    for _ in 0..FRAMES {
        world.tick(true);
        frame_pixels.fill(0);
        for pellet in world.pellets() {
            Renderer::draw_pellet(pellet, &mut frame_pixels, WIDTH, HEIGHT);
        }
        for critter in world.critters() {
            Renderer::draw(critter, CRITTER_RADIUS, &mut frame_pixels, WIDTH, HEIGHT);
        }
    }

    // Critters might split (more) or run out of energy and stop ticking, but the
    // world should never lose all of them in 1000 frames from a fresh start.
    assert!(!world.critters().is_empty());
    // The frame buffer remains the right size.
    assert_eq!(frame_pixels.len(), WIDTH * HEIGHT);
}
