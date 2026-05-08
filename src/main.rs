use circles::{text_pixels, Renderer, StagnationDetector, World, CRITTER_RADIUS};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use rand::thread_rng;

const FRAME_DURATION_MICROSECONDS: u64 = 16_667;
const BACKGROUND_COLOR: u32 = 0x00_00_00;
const TEXT_COLOR: u32 = 0xFF_FF_FF;
const TEXT_SCALE: usize = 4;
const TEXT_MARGIN: usize = 16;
const ENERGY_REFRESH_FRAMES: u32 = 30;
const STAGNATION_THRESHOLD_FRAMES: u32 = 300;

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
    let mut frame_counter: u32 = 0;
    let mut displayed_total_energy = world.total_energy();
    let mut stagnation = StagnationDetector::new(STAGNATION_THRESHOLD_FRAMES);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            world.reset(&mut rng);
            stagnation.reset();
        }

        world.tick();

        let total_energy = world.total_energy();
        stagnation.observe(total_energy);
        if stagnation.is_stagnant() {
            world.reset(&mut rng);
            stagnation.reset();
        }

        if frame_counter.is_multiple_of(ENERGY_REFRESH_FRAMES) {
            displayed_total_energy = total_energy;
        }
        frame_counter = frame_counter.wrapping_add(1);

        frame_pixels.fill(BACKGROUND_COLOR);
        for pellet in world.pellets() {
            Renderer::draw_pellet(pellet, &mut frame_pixels, width, height);
        }
        for critter in world.critters() {
            Renderer::draw(critter, CRITTER_RADIUS, &mut frame_pixels, width, height);
        }
        draw_total_energy(displayed_total_energy, &mut frame_pixels, width, height);

        window
            .update_with_buffer(&frame_pixels, width, height)
            .expect("Unable to update window");
    }
}

fn draw_total_energy(value: u32, buffer: &mut [u32], width: usize, height: usize) {
    let text = value.to_string();
    let pixels = text_pixels(&text, TEXT_SCALE);
    let text_width = pixels.iter().map(|&(x, _)| x + 1).max().unwrap_or(0);
    let origin_x = width.saturating_sub(text_width + TEXT_MARGIN);
    let origin_y = TEXT_MARGIN;
    for (x, y) in pixels {
        let px = origin_x + x;
        let py = origin_y + y;
        if px < width && py < height {
            buffer[py * width + px] = TEXT_COLOR;
        }
    }
}
