//! Keeps logs that a test triggers on purpose out of the test output.
//!
//! Some tests give a filter invalid input to check that it refuses cleanly,
//! which logs a warning or error. `expect_logs()` drops log output from the
//! calling test's thread while the returned guard is alive. Logs from every
//! other test still print, so unexpected warnings stay visible.

use std::cell::Cell;
use std::sync::{Arc, Once};

use spdlog::formatter::Formatter;
use spdlog::sink::Sink;
use spdlog::{ErrorHandler, LevelFilter, Logger, Record};

thread_local! {
    static QUIET: Cell<bool> = const { Cell::new(false) };
}

/// Forwards records to the original default logger unless the logging
/// thread is quiet.
struct QuietableSink {
    inner: Arc<Logger>,
}

impl Sink for QuietableSink {
    fn log(&self, record: &Record) -> spdlog::Result<()> {
        if !QUIET.with(Cell::get) {
            self.inner.log(record);
        }
        Ok(())
    }

    fn flush(&self) -> spdlog::Result<()> {
        self.inner.flush();
        Ok(())
    }

    fn level_filter(&self) -> LevelFilter {
        LevelFilter::All
    }

    fn set_level_filter(&self, _level_filter: LevelFilter) {}

    fn set_formatter(&self, _formatter: Box<dyn Formatter>) {}

    fn set_error_handler(&self, _handler: Option<ErrorHandler>) {}
}

fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let inner = spdlog::default_logger();
        let logger = Logger::builder()
            .level_filter(LevelFilter::All)
            .sink(Arc::new(QuietableSink { inner }))
            .build()
            .expect("test logger");
        spdlog::set_default_logger(Arc::new(logger));
    });
}

/// Silences log output from this test while alive.
pub struct ExpectedLogs;

impl Drop for ExpectedLogs {
    fn drop(&mut self) {
        QUIET.with(|quiet| quiet.set(false));
    }
}

/// For tests that trigger warnings or errors on purpose: hold the guard for
/// the rest of the test, e.g. `let _quiet = expect_logs();`.
#[must_use = "logs are only silenced while the guard is alive"]
pub fn expect_logs() -> ExpectedLogs {
    install();
    QUIET.with(|quiet| quiet.set(true));
    ExpectedLogs
}
