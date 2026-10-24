//! Routes Rust-side `spdlog` messages into Python's standard `logging` module,
//! under the logger name `determined`, so they can be filtered, captured and
//! redirected like any other Python log output.

use std::sync::Arc;

use pyo3::prelude::*;
use spdlog::formatter::Formatter;
use spdlog::sink::Sink;
use spdlog::{ ErrorHandler, Level, LevelFilter, Logger, Record };

const LOGGER_NAME: &str = "determined";

/// Python `logging` level numbers.
fn py_level(level: Level) -> u32 {
    match level {
        Level::Critical => 50,
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    }
}

/// spdlog sink that forwards every record to `logging.getLogger("determined")`.
///
/// Filtering and formatting are left to Python, so the level filter,
/// formatter and error handler are intentionally ignored.
struct PyLogSink;

impl Sink for PyLogSink {
    fn log(&self, record: &Record) -> spdlog::Result<()> {
        Python::attach(|py| {
            // never fail the caller because of a logging problem
            let _ = py.import("logging")
                .and_then(|logging| logging.call_method1("getLogger", (LOGGER_NAME,)))
                .and_then(|logger| logger.call_method1("log", (py_level(record.level()), record.payload())));
        });
        Ok(())
    }

    fn flush(&self) -> spdlog::Result<()> {
        Ok(())
    }

    fn level_filter(&self) -> LevelFilter {
        LevelFilter::All
    }

    fn set_level_filter(&self, _level_filter: LevelFilter) {}

    fn set_formatter(&self, _formatter: Box<dyn Formatter>) {}

    fn set_error_handler(&self, _handler: Option<ErrorHandler>) {}
}

/// Make `spdlog` use Python's `logging`. Called once on module import.
pub fn install() -> PyResult<()> {
    let logger = Logger::builder()
        .name(LOGGER_NAME)
        .level_filter(LevelFilter::All)
        .sink(Arc::new(PyLogSink))
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    spdlog::set_default_logger(Arc::new(logger));
    Ok(())
}
