use rand::Rng;

pub fn random_choice(size: usize, max: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();

    let mut choice = Vec::<usize>::with_capacity(size as usize);
    for _i in 0..size {
        choice.push((rng.gen::<f32>()*max as f32).floor() as usize);
    }
    
    return choice;
}