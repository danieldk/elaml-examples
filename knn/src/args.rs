use clap::{App, AppSettings, Arg, ArgMatches};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub fn parse_args() -> ArgMatches<'static> {
    App::new("final-frontier")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name("knearest")
                .short("k")
                .long("knearest")
                .value_name("K")
                .help("Number of nearest neighbors to consider in voting: (default: 3)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("TRAIN")
                .help("Train data")
                .index(1)
                .required(true),
        )
        .arg(
            Arg::with_name("TEST")
                .help("Test data")
                .index(2)
                .required(true),
        )
        .get_matches()
}
