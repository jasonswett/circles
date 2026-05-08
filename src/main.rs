use circles::{text_pixels, FpsCounter, Renderer, StagnationDetector, World, CRITTER_RADIUS};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use rand::thread_rng;
use std::time::{Duration, Instant};

const FRAME_DURATION_MICROSECONDS: u64 = 16_667;
const BACKGROUND_COLOR: u32 = 0x00_00_00;
const TEXT_COLOR: u32 = 0xFF_FF_FF;
const TEXT_SIZE: f32 = 28.0;
const TEXT_LINE_HEIGHT: usize = 36;
const TEXT_MARGIN: usize = 16;
const ENERGY_REFRESH_FRAMES: u32 = 30;
const STAGNATION_THRESHOLD_FRAMES: u32 = 300;
const REAPER_INTERVAL_FRAMES: u32 = 300;
const REPLENISH_INTERVAL_FRAMES: u32 = 300;
const REPLENISH_MIN_FPS: u32 = 40;
const FPS_REFRESH_INTERVAL: Duration = Duration::from_millis(500);

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
    let mut fps_counter = FpsCounter::new(FPS_REFRESH_INTERVAL);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        fps_counter.observe_frame(Instant::now());
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
        if frame_counter > 0 && frame_counter.is_multiple_of(REAPER_INTERVAL_FRAMES) {
            world.reap_dead_critters();
        }
        if frame_counter > 0
            && frame_counter.is_multiple_of(REPLENISH_INTERVAL_FRAMES)
            && fps_counter.current_fps() >= REPLENISH_MIN_FPS
        {
            world.replenish_pellets(&mut rng);
        }
        frame_counter = frame_counter.wrapping_add(1);

        frame_pixels.fill(BACKGROUND_COLOR);
        for pellet in world.pellets() {
            Renderer::draw_pellet(pellet, &mut frame_pixels, width, height);
        }
        for critter in world.critters() {
            Renderer::draw(critter, CRITTER_RADIUS, &mut frame_pixels, width, height);
        }
        let energy_text = format!("Energy: {displayed_total_energy}");
        let fps_text = format!("FPS: {}", fps_counter.current_fps());
        let population_text = format!("Population: {}", world.critters().len());
        draw_text_top_right(&energy_text, 0, &mut frame_pixels, width, height);
        draw_text_top_right(&fps_text, 1, &mut frame_pixels, width, height);
        draw_text_top_right(&population_text, 2, &mut frame_pixels, width, height);

        window
            .update_with_buffer(&frame_pixels, width, height)
            .expect("Unable to update window");
    }
}

fn draw_text_top_right(
    text: &str,
    line_index: usize,
    buffer: &mut [u32],
    width: usize,
    height: usize,
) {
    let pixels = text_pixels(text, TEXT_SIZE);
    let text_width = pixels.iter().map(|&(x, _, _)| x + 1).max().unwrap_or(0);
    let origin_x = width.saturating_sub(text_width + TEXT_MARGIN);
    let origin_y = TEXT_MARGIN + line_index * TEXT_LINE_HEIGHT;
    for (x, y, alpha) in pixels {
        let px = origin_x + x;
        let py = origin_y + y;
        if px < width && py < height {
            let existing = buffer[py * width + px];
            buffer[py * width + px] = blend(existing, TEXT_COLOR, alpha);
        }
    }
}

fn blend(background: u32, foreground: u32, alpha: u8) -> u32 {
    let a = alpha as u32;
    let inv = 255 - a;
    let bg_r = (background >> 16) & 0xFF;
    let bg_g = (background >> 8) & 0xFF;
    let bg_b = background & 0xFF;
    let fg_r = (foreground >> 16) & 0xFF;
    let fg_g = (foreground >> 8) & 0xFF;
    let fg_b = foreground & 0xFF;
    let r = (fg_r * a + bg_r * inv) / 255;
    let g = (fg_g * a + bg_g * inv) / 255;
    let b = (fg_b * a + bg_b * inv) / 255;
    (r << 16) | (g << 8) | b
}
