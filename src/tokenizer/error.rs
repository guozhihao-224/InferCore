use snafu::{Location, Snafu};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("tokenizers error"))]
    Tokenizer {
        #[snafu(source)]
        error: tokenizers::Error,
        #[snafu(implicit)]
        location: Location,
    },
}
