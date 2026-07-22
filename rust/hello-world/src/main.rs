use rand::RngExt;

fn main() {
    println!("Hello World !");

    let mut rng = rand::rng();
    let i: i32 = rng.random();
    let f: f32 = rng.random();
    println!("random  int  number = {}", i);
    println!("random float number = {}", f);
}

