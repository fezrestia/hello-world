use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use fancy_regex::Regex;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use log::info;

fn pretokenize(text: &str) -> Vec<String> {
    static PATTERN: &str =
        r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;  // raw string

    let re = Regex::new(PATTERN).unwrap();

    return re.find_iter(text)
            .map(|m| {
                m.unwrap().as_str().to_string()
            })
            .collect();
}

#[pyfunction]
fn proc_pretoken_chunk(
        file_path: &str,
        start: u64,
        end: u64,
        end_token: &str,
) -> HashMap<String, i32> {
    let mut pretoken_vs_count: HashMap<String, i32> = HashMap::new();

    let mut f = File::open(file_path).unwrap();

    f.seek(SeekFrom::Start(start)).unwrap();  // move cursor to start
    let mut chunk_bytes = vec![0u8; (end - start) as usize];  // read buffer
    f.read_exact(&mut chunk_bytes).unwrap();  // read

    let chunk_text = String::from_utf8_lossy(&chunk_bytes);

    for text in chunk_text.split(end_token) {
        for pretoken in pretokenize(text) {
            *pretoken_vs_count.entry(pretoken).or_insert(0) += 1;
        }
    }

    return pretoken_vs_count;
}



fn count_pairs(
        ids: &Vec<i32>,
        weight: i32,
        counts: Option<HashMap<(i32, i32), i32>>,
) -> HashMap<(i32, i32), i32> {
    let mut counts = counts.unwrap_or_default();

    for pair in ids.windows(2) {
        let key = (pair[0], pair[1]);
        *counts.entry(key).or_insert(0) += weight;
    }

    return counts
}

fn merge(
        ids: &Vec<i32>,
        pair: (i32, i32),
        new_id: i32,
) -> Vec<i32> {
    let mut merged_ids: Vec<i32> = Vec::new();

    let mut i = 0;

    while i < ids.len() {
        if i < ids.len() - 1 && (ids[i], ids[i + 1]) == pair {
            // hit pair, convert to new_id
            merged_ids.push(new_id);
            i += 2;
        } else {
            // no pair, stay single char
            merged_ids.push(ids[i]);
            i += 1;
        }
    }

    return merged_ids
}

#[pyfunction]
fn train_bpe_loop(
        num_merges: i32,
        mut pair_vs_count: HashMap<(i32, i32), i32>,
        mut pair_vs_ids: HashMap<(i32, i32), HashSet<Vec<i32>>>,
        mut ids_vs_count: HashMap<Vec<i32>, i32>,
) -> Vec<Vec<i32>> {
    println!("pyo3.train_bpe_loop() : E");

    let log_steps: i32 = (num_merges / 100).max(1);

    let mut elapsed_total_ms = Duration::ZERO;
    let mut elapsed_count: u32 = 0;

    let mut merge_rules: Vec<Vec<i32>> = Vec::new();

    for step in 0..num_merges {
        if pair_vs_count.is_empty() {
            // NOP, there is no pair
            break
        }

        let start_ts = Instant::now();

        let (&most_available_pair, _) = pair_vs_count
                .iter()
                .max_by_key(|(pair, count)| (*count, pair.0, pair.1))
                .unwrap();

        let new_id: i32 = 256 + step;
        let (id_1, id_2) = most_available_pair;
        merge_rules.push(vec![id_1, id_2, new_id]);

        let affected_ids = pair_vs_ids.remove(&most_available_pair).unwrap();

        for ids in affected_ids {
            let ids_count = *ids_vs_count.get(&ids).unwrap();

            let new_ids = merge(&ids, most_available_pair, new_id);

            ids_vs_count.remove(&ids);
            ids_vs_count.insert(new_ids.clone(), ids_count);


            // update old
            let old_pair_vs_count = count_pairs(&ids, 1, None);
            for (pair, count) in old_pair_vs_count {
                if let Some(v) = pair_vs_count.get_mut(&pair) {
                    *v -= count * ids_count;

                    if *v <= 0 {
                        pair_vs_count.remove(&pair);
                    }
                }

                if let Some(s) = pair_vs_ids.get_mut(&pair) {
                    s.remove(&ids);
                }
            }

            // update new
            let new_pair_vs_count = count_pairs(&new_ids, 1, None);
            for (pair, count) in new_pair_vs_count {
                *pair_vs_count.entry(pair).or_insert(0) += count * ids_count;

                pair_vs_ids
                    .entry(pair)
                    .or_insert_with(HashSet::new)
                    .insert(new_ids.clone());
            }
        }

        elapsed_total_ms += start_ts.elapsed();
        elapsed_count += 1;

        if step % log_steps == 0 {
            let log_line = format!("train_bpe_loop ... {}/{} [{} ms in {} loops]",
                    step,
                    num_merges,
                    elapsed_total_ms.as_secs_f64() * 1000.0,
                    elapsed_count,
            );

            println!("{}", log_line);
            info!("{}", log_line);

            elapsed_total_ms = Duration::ZERO;
            elapsed_count = 0;
        }
    }

    println!("pyo3.train_bpe_loop() : X");
    return merge_rules
}



#[pymodule]
fn tokenizer_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(proc_pretoken_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe_loop, m)?)?;
    Ok(())
}

