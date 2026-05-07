use circles::{Critter, Heading, Instruction, Renderer};
use minifb::{Key, Window, WindowOptions};
use rand::thread_rng;
use rand::Rng;

const FRAME_DURATION_MICROSECONDS: u64 = 16_667;
const TICKS_PER_INSTRUCTION: u32 = 15;
const INSTRUCTION_LIST_LENGTH: usize = 4;
const CRITTER_RADIUS: i32 = 20;
const STEP_SIZE: i32 = 25;
const NUM_CRITTERS: usize = 8;
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

fn make_critter<R: Rng>(rng: &mut R, width: usize, height: usize) -> Critter {
    let instructions = Instruction::random_list(rng, INSTRUCTION_LIST_LENGTH);
    let x = rng.gen_range(CRITTER_RADIUS..(width as i32 - CRITTER_RADIUS));
    let y = rng.gen_range(CRITTER_RADIUS..(height as i32 - CRITTER_RADIUS));
    Critter::new(
        x,
        y,
        Heading::random(rng),
        instructions,
        TICKS_PER_INSTRUCTION,
        STEP_SIZE,
    )
}

fn make_critters(width: usize, height: usize) -> Vec<Critter> {
    let mut rng = thread_rng();
    (0..NUM_CRITTERS)
        .map(|_| make_critter(&mut rng, width, height))
        .collect()
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

    let mut critters = make_critters(width, height);
    let mut frame_pixels = vec![BACKGROUND_COLOR; width * height];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_down(Key::Space) {
            critters = make_critters(width, height);
        }

        for critter in &mut critters {
            critter.tick();
        }

        frame_pixels.fill(BACKGROUND_COLOR);
        for critter in &critters {
            Renderer::draw(critter, CRITTER_RADIUS, &mut frame_pixels, width, height);
        }

        window
            .update_with_buffer(&frame_pixels, width, height)
            .expect("Unable to update window");
    }
}
