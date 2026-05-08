use circles::{Renderer, World, CRITTER_RADIUS};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use rand::thread_rng;

const FRAME_DURATION_MICROSECONDS: u64 = 16_667;
const BACKGROUND_COLOR: u32 = 0x00_00_00;

#[repr(C)]
struct CGSize {
    width: f64,
    height: f64,
}

#[repr(C)]
struct CGRect {
    _origin_x: f64,
    _origin_y: f64,
    size: CGSize,
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGMainDisplayID() -> u32;
    fn CGDisplayBounds(display: u32) -> CGRect;
}

fn main() {
    let (width, height) = unsafe {
        let display = CGMainDisplayID();
        let bounds = CGDisplayBounds(display);
        (bounds.size.width as usize, bounds.size.height as usize)
    };

    let mut window = Window::new(
        "Circles",
        width,
        height,
        WindowOptions {
            borderless: true,
            title: false,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to create window");

    window.limit_update_rate(Some(std::time::Duration::from_micros(
        FRAME_DURATION_MICROSECONDS,
    )));

    let mut rng = thread_rng();
    let mut world = World::new(width, height, &mut rng);
    let mut frame_pixels = vec![BACKGROUND_COLOR; width * height];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            world.reset(&mut rng);
        }

        world.tick();

        frame_pixels.fill(BACKGROUND_COLOR);
        for pellet in world.pellets() {
            Renderer::draw_pellet(pellet, &mut frame_pixels, width, height);
        }
        for critter in world.critters() {
            Renderer::draw(critter, CRITTER_RADIUS, &mut frame_pixels, width, height);
        }

        window
            .update_with_buffer(&frame_pixels, width, height)
            .expect("Unable to update window");
    }
}
