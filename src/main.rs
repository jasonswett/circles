use circles::{Critter, Heading, Instruction, Renderer};
use minifb::{Key, Window, WindowOptions};
use rand::thread_rng;

const FRAME_DURATION_MICROSECONDS: u64 = 16_667;
const TICKS_PER_INSTRUCTION: u32 = 30;
const INSTRUCTION_LIST_LENGTH: usize = 32;
const CRITTER_RADIUS: i32 = 20;
const STEP_SIZE: i32 = 50;
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

fn make_critter(width: usize, height: usize) -> Critter {
    let mut rng = thread_rng();
    let instructions = Instruction::random_list(&mut rng, INSTRUCTION_LIST_LENGTH);
    Critter::new(
        (width / 2) as i32,
        (height / 2) as i32,
        Heading::North,
        instructions,
        TICKS_PER_INSTRUCTION,
        STEP_SIZE,
    )
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

    window.limit_update_rate(Some(std::time::Duration::from_micros(FRAME_DURATION_MICROSECONDS)));

    let mut critter = make_critter(width, height);
    let mut frame_pixels = vec![BACKGROUND_COLOR; width * height];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_down(Key::Space) {
            critter = make_critter(width, height);
        }

        critter.tick();

        frame_pixels.fill(BACKGROUND_COLOR);
        Renderer::draw(&critter, CRITTER_RADIUS, &mut frame_pixels, width, height);

        window
            .update_with_buffer(&frame_pixels, width, height)
            .expect("Unable to update window");
    }
}
