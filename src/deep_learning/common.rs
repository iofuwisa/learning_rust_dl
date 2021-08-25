use rand::Rng;

pub fn random_choice(size: u32, max: u32) -> Vec<u32> {
    let mut rng = rand::thread_rng();

    let mut choice = Vec::<u32>::with_capacity(size as usize);
    for _i in 0..size {
        choice.push((rng.gen::<f32>()*max as f32).floor() as u32);
    }
    
    return choice;
}