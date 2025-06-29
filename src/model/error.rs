use snafu::Location;
use snafu::Snafu;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("mlp error"))]
    MlpErr {
        #[snafu(source)]
        error: candle_core::Error,
        #[snafu(implicit)]
        location: Location,
    },
}
