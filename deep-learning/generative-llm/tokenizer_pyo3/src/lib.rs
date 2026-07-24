use pyo3::prelude::*;use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
//use fancy_regex::Regex;
use regex::Regex;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use log::info;
use pyo3::types::{PyDict, PyTuple, PySet};
use once_cell::sync::Lazy;


static REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        //r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"#
    ).unwrap()
});

fn pretokenize(text: &str) -> Vec<String> {
//    static PATTERN: &str =
//        r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;  // raw string
//
//    let re = Regex::new(PATTERN).unwrap();
//
//    return re.find_iter(text)

//    return REGEX.find_iter(text)
//            .map(|m| {
//                m.unwrap().as_str().to_string()
//            })
//            .collect();

    return REGEX.find_iter(text)
            .map(|m| {
                m.as_str().to_string()
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
    info!("pyo3.proc_pretoken_chunk() : E / [{}-{}]", start, end);
    println!("pyo3.proc_pretoken_chunk() : E / [{}-{}]", start, end);


    let mut start_ts = Instant::now();


    let mut pretoken_vs_count: HashMap<String, i32> = HashMap::new();

    let mut f = File::open(file_path).unwrap();


    info!("pyo3.proc_pretoken_chunk() : File::open() done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    println!("pyo3.proc_pretoken_chunk() : File::open() done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    start_ts = Instant::now();


    f.seek(SeekFrom::Start(start)).unwrap();  // move cursor to start


    info!("pyo3.proc_pretoken_chunk() : seek() done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    println!("pyo3.proc_pretoken_chunk() : seek() done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    start_ts = Instant::now();


    let mut chunk_bytes = vec![0u8; (end - start) as usize];  // read buffer


    info!("pyo3.proc_pretoken_chunk() : chunk bytes done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    println!("pyo3.proc_pretoken_chunk() : chunk bytes done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    start_ts = Instant::now();


    f.read_exact(&mut chunk_bytes).unwrap();  // read


    info!("pyo3.proc_pretoken_chunk() : read buf done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    println!("pyo3.proc_pretoken_chunk() : read buf done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    start_ts = Instant::now();


    let chunk_text = String::from_utf8_lossy(&chunk_bytes);


    info!("pyo3.proc_pretoken_chunk() : from_utf8_lossy done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    println!("pyo3.proc_pretoken_chunk() : from_utf8_lossy done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    start_ts = Instant::now();


    for text in chunk_text.split(end_token) {
        for pretoken in pretokenize(text) {
            *pretoken_vs_count.entry(pretoken).or_insert(0) += 1;
        }
    }


    info!("pyo3.proc_pretoken_chunk() : pretokenize for x2 done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    println!("pyo3.proc_pretoken_chunk() : pretokenize for x2 done [{} ms]", start_ts.elapsed().as_secs_f64() * 1000.0);
    //start_ts = Instant::now();


    info!("pyo3.proc_pretoken_chunk() : X");
    println!("pyo3.proc_pretoken_chunk() : X");
    return pretoken_vs_count;
}



#[pyfunction]
fn encode_pretoken(
        py: Python,
        pretoken_vs_count: HashMap<String, i32>,
) -> PyResult<Py<PyDict>> {  // dict[tuple[int, ...], int]
    let py_dict = PyDict::new(py);

    for (pretoken, count) in pretoken_vs_count {
        // String -> utf-8 bytes -> Vec<i32>
        let ids: Vec<i32> = pretoken .into_bytes().into_iter().map(|b| b as i32).collect();

        // Vec<i32> -> PyTuple
        let key = PyTuple::new(py, ids)?;

        py_dict.set_item(key, count)?;
    }

    return Ok(py_dict.unbind());  // unbound from py
}


fn count_pairs(
        ids: &Vec<i32>,
        weight: i32,
        counts: &mut HashMap<(i32, i32), i32>,
) {
    for pair in ids.windows(2) {
        let key = (pair[0], pair[1]);
        *counts.entry(key).or_insert(0) += weight;
    }
}


#[pyfunction]
fn gen_cache(
        py: Python,
        ids_vs_count: HashMap<Vec<i32>, i32>,
) -> PyResult<(Py<PyDict>, Py<PyDict>)> {  // pair_vs_count, pair_vs_ids
    let mut pair_vs_count_map: HashMap<(i32, i32), i32> = HashMap::new();
    let pair_vs_ids = PyDict::new(py);

    for (ids, count) in ids_vs_count.iter() {
        count_pairs(ids, *count, &mut pair_vs_count_map);

        for pair in ids.windows(2).map(|w| (w[0], w[1])) {  // [0, 1, 2, 3] -> [0, 1], [1, 2], ...
            let ids_set = match pair_vs_ids.get_item(pair)? {
                Some(obj) => obj.cast_into::<PySet>()?,
                None => {
                    let s = PySet::empty(py)?;
                    pair_vs_ids.set_item(pair, &s)?;
                    s
                }
            };

            let val = PyTuple::new(py, ids)?;
            ids_set.add(val)?;
        }
    }

    let pair_vs_count = PyDict::new(py);
    for (pair, count) in pair_vs_count_map.iter() {
        pair_vs_count.set_item(pair, count)?;
    }

    return Ok((pair_vs_count.unbind(), pair_vs_ids.unbind()));
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
            let mut old_pair_vs_count: HashMap<(i32, i32), i32> = HashMap::new();
            count_pairs(&ids, 1, &mut old_pair_vs_count);
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
            let mut new_pair_vs_count: HashMap<(i32, i32), i32> = HashMap::new();
            count_pairs(&new_ids, 1, &mut new_pair_vs_count);
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
    m.add_function(wrap_pyfunction!(encode_pretoken, m)?)?;
    m.add_function(wrap_pyfunction!(gen_cache, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe_loop, m)?)?;
    Ok(())
}

