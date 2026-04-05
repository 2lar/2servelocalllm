use metrics::counter;

pub fn record_cache_hit() {
    counter!("llm_cache_hits_total").increment(1);
}

pub fn record_cache_miss() {
    counter!("llm_cache_misses_total").increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_functions_do_not_panic() {
        record_cache_hit();
        record_cache_miss();
    }
}
