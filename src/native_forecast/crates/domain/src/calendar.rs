//! The proleptic Gregorian calendar on timestamps in nanoseconds since
//! 1970-01-01, and TempoPFN's sampling frequencies with the rules by which it
//! picks a frequency and a start date for a synthetic series
//! (`src/data/frequency.py` and `src/data/constants.py` of TempoPFN).

use crate::random::Rng;

/// The nanoseconds of a day.
pub const DAY: i64 = 86_400_000_000_000;
const MINUTE: i64 = 60_000_000_000;

/// The days from 1970-01-01 to the given date (Hinnant's `days_from_civil`).
#[must_use]
pub const fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let shifted_month = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The year, month and day of the given number of days from 1970-01-01
/// (Hinnant's `civil_from_days`).
#[must_use]
pub const fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + if month <= 2 { 1 } else { 0 };
    (year, month, day)
}

/// The calendar fields of a timestamp that the seasonal components read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fields {
    /// The month, from 1.
    pub month: i64,
    /// The day of the month, from 1.
    pub day: i64,
    /// The day of the week, Monday 0.
    pub weekday: i64,
    /// The hour of the day.
    pub hour: i64,
    /// The minute of the hour.
    pub minute: i64,
}

/// The calendar fields of the timestamp `ns`.
#[must_use]
pub const fn fields(ns: i64) -> Fields {
    let days = ns.div_euclid(DAY);
    let in_day = ns.rem_euclid(DAY);
    let (_, month, day) = civil_from_days(days);
    Fields {
        month,
        day,
        // 1970-01-01 was a Thursday.
        weekday: (days + 3).rem_euclid(7),
        hour: in_day / (60 * MINUTE),
        minute: in_day / MINUTE % 60,
    }
}

/// A sampling frequency of TempoPFN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Frequency {
    /// Yearly.
    A,
    /// Quarterly.
    Q,
    /// Monthly.
    M,
    /// Weekly, on Sundays.
    W,
    /// Daily.
    D,
    /// Hourly.
    H,
    /// Every second.
    S,
    /// Every minute.
    T1,
    /// Every 5 minutes.
    T5,
    /// Every 10 minutes.
    T10,
    /// Every 15 minutes.
    T15,
    /// Every 30 minutes.
    T30,
}

impl Frequency {
    /// Every frequency, in the order of TempoPFN's enumeration.
    pub const ALL: [Self; 12] = [
        Self::A,
        Self::Q,
        Self::M,
        Self::W,
        Self::D,
        Self::H,
        Self::S,
        Self::T1,
        Self::T5,
        Self::T10,
        Self::T15,
        Self::T30,
    ];

    /// The days of a period: TempoPFN's `FREQUENCY_MAPPING`.
    #[must_use]
    pub const fn days_per_period(self) -> f64 {
        match self {
            Self::A => 365.25,
            Self::Q => 91.3125,
            Self::M => 30.4375,
            Self::W => 7.0,
            Self::D => 1.0,
            Self::H => 1.0 / 24.0,
            Self::S => 1.0 / 86_400.0,
            Self::T1 => 1.0 / 1440.0,
            Self::T5 => 1.0 / 288.0,
            Self::T10 => 1.0 / 144.0,
            Self::T15 => 1.0 / 96.0,
            Self::T30 => 1.0 / 48.0,
        }
    }

    /// The nanoseconds of a period, for the frequencies of a fixed step.
    #[must_use]
    pub const fn step(self) -> Option<i64> {
        match self {
            Self::A | Self::Q | Self::M => None,
            Self::W => Some(7 * DAY),
            Self::D => Some(DAY),
            Self::H => Some(60 * MINUTE),
            Self::S => Some(MINUTE / 60),
            Self::T1 => Some(MINUTE),
            Self::T5 => Some(5 * MINUTE),
            Self::T10 => Some(10 * MINUTE),
            Self::T15 => Some(15 * MINUTE),
            Self::T30 => Some(30 * MINUTE),
        }
    }

    /// The weight of the frequency in GIFT-Eval:
    /// `GIFT_EVAL_FREQUENCY_WEIGHTS`.
    const fn weight(self) -> f64 {
        match self {
            Self::H => 25.0,
            Self::D => 23.4,
            Self::W => 12.9,
            Self::T15 | Self::T5 => 9.7,
            Self::M => 7.3,
            Self::T10 | Self::S => 4.8,
            Self::T1 => 1.6,
            Self::Q | Self::A => 0.8,
            Self::T30 => 0.1,
        }
    }

    /// The least and greatest length of a series of the frequency and the
    /// ends of its preferred range: `GIFT_EVAL_LENGTH_RANGES`.
    const fn length_range(self) -> (f64, f64, f64, f64) {
        match self {
            Self::A => (25.0, 100.0, 30.0, 70.0),
            Self::Q => (25.0, 150.0, 50.0, 120.0),
            Self::M => (40.0, 1000.0, 100.0, 600.0),
            Self::W => (50.0, 3500.0, 100.0, 1500.0),
            Self::D => (150.0, 25_000.0, 300.0, 7000.0),
            Self::H => (600.0, 35_000.0, 700.0, 17_000.0),
            Self::T1 => (200.0, 2500.0, 1200.0, 1800.0),
            Self::S => (7500.0, 9500.0, 7900.0, 9000.0),
            Self::T15 => (1000.0, 140_000.0, 50_000.0, 130_000.0),
            Self::T5 => (200.0, 105_000.0, 20_000.0, 95_000.0),
            Self::T10 => (40_000.0, 55_000.0, 47_000.0, 52_000.0),
            Self::T30 => (100.0, 50_000.0, 10_000.0, 40_000.0),
        }
    }

    /// The greatest length that stays within 500 years:
    /// `ALL_FREQUENCY_MAX_LENGTHS`.
    fn max_length(self) -> f64 {
        const YEARS: f64 = 500.0;
        match self {
            Self::A => YEARS,
            Self::Q => YEARS * 4.0,
            Self::M => YEARS * 12.0,
            Self::W => (YEARS * 52.1775).floor(),
            other => (YEARS * 365.2425 / other.days_per_period()).floor(),
        }
    }

    /// The weight with which `select_safe_random_frequency` draws the
    /// frequency for a series of `length` steps; 0 if it never does.
    #[must_use]
    pub fn score(self, length: usize) -> f64 {
        let length = length as f64;
        let (low, high, preferred_low, preferred_high) = self.length_range();
        if length > self.max_length() || length < low || length > high {
            return 0.0;
        }
        let multiplier = if (preferred_low..=preferred_high).contains(&length) {
            5.0
        } else {
            let distance = if length < preferred_low {
                (preferred_low - length) / (preferred_low - low)
            } else {
                (length - preferred_high) / (high - preferred_high)
            };
            0.3 + 1.2 * (1.0 - distance)
        };
        self.weight() * multiplier
    }

    /// A frequency for a series of `length` steps, drawn as TempoPFN's
    /// `select_safe_random_frequency` draws it. Lengths that no frequency
    /// suits fall back on the first of TempoPFN's fallback orders.
    pub fn select(rng: &mut Rng, length: usize) -> Self {
        let scores = Self::ALL.map(|frequency| frequency.score(length));
        if scores.iter().any(|&score| score > 0.0) {
            Self::ALL[rng.weighted(&scores)]
        } else if length <= 100 {
            Self::A
        } else if length <= 1000 {
            Self::D
        } else {
            Self::H
        }
    }

    /// A start for a series of `length` steps of this frequency, uniform
    /// between 1700-01-01 and the last instant that keeps the series before
    /// 2200-01-01: TempoPFN's `select_safe_start_date`.
    pub fn start(self, rng: &mut Rng, length: usize) -> i64 {
        let earliest = days_from_civil(1700, 1, 1) * DAY;
        let span = (length as f64 * self.days_per_period()) as i64;
        let latest = (days_from_civil(2200, 1, 1) - span) * DAY;
        rng.integer(earliest, latest.max(earliest) + 1)
    }

    /// Moves `start` back to the last instant on the grid of the frequency,
    /// as the `rollback` of the pandas offset does: to a Sunday for weeks,
    /// nowhere for the frequencies of a fixed number of seconds.
    #[must_use]
    pub const fn rollback(self, start: i64) -> i64 {
        match self {
            Self::W => {
                let weekday = fields(start).weekday;
                start - (weekday + 1) % 7 * DAY
            }
            _ => start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_convert_both_ways() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(days_from_civil(1700, 1, 1), -98_615);
        assert_eq!(civil_from_days(-98_615), (1700, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        for days in (-100_000..90_000).step_by(37) {
            let (year, month, day) = civil_from_days(days);
            assert_eq!(days_from_civil(year, month, day), days);
        }
    }

    #[test]
    fn fields_follow_the_calendar() {
        // 2024-02-29 13:47 was a Thursday.
        let ns = 19_782 * DAY + (13 * 60 + 47) * MINUTE + 5;
        let expected = Fields {
            month: 2,
            day: 29,
            weekday: 3,
            hour: 13,
            minute: 47,
        };
        assert_eq!(fields(ns), expected);
        // 1699-12-31 23:59, a Thursday, before the epoch.
        let before = days_from_civil(1700, 1, 1) * DAY - MINUTE;
        let expected = Fields {
            month: 12,
            day: 31,
            weekday: 3,
            hour: 23,
            minute: 59,
        };
        assert_eq!(fields(before), expected);
    }

    #[test]
    fn frequencies_of_a_corpus_series_follow_the_scores() {
        // For 2048 steps TempoPFN's scores are 5 × 25.0 for H, 5 × 23.4 for
        // D, and below 16 for W, T1, T5, T15 and T30; the others are out of
        // their length ranges.
        let scores = Frequency::ALL.map(|frequency| frequency.score(2048));
        assert_eq!(scores[5], 125.0);
        assert_eq!(scores[4], 117.0);
        assert!((scores[3] - 12.9 * (0.3 + 1.2 * (1.0 - 548.0 / 2000.0))).abs() < 1e-12);
        for index in [0, 1, 2, 6, 9] {
            assert_eq!(scores[index], 0.0, "{:?}", Frequency::ALL[index]);
        }
        let total: f64 = scores.iter().sum();
        assert!((total - 266.04).abs() < 0.01, "{total}");
        let mut rng = Rng::from_key(&[3]);
        let draws = 20_000;
        let hourly = (0..draws)
            .filter(|_| Frequency::select(&mut rng, 2048) == Frequency::H)
            .count();
        assert!((hourly as f64 / f64::from(draws) - 125.0 / total).abs() < 0.015);
        assert_eq!(Frequency::select(&mut rng, 20), Frequency::A);
    }

    #[test]
    fn starts_keep_the_series_before_2200_and_weeks_start_on_sunday() {
        let mut rng = Rng::from_key(&[4]);
        let first = days_from_civil(1700, 1, 1) * DAY;
        let last = days_from_civil(2200, 1, 1) * DAY;
        for frequency in [Frequency::W, Frequency::D, Frequency::H, Frequency::T5] {
            for _ in 0..500 {
                let start = frequency.start(&mut rng, 2048);
                let end = start + 2047 * frequency.step().expect("a fixed step");
                assert!(start >= first && end <= last);
                let rolled = frequency.rollback(start);
                assert!(rolled <= start && start - rolled < 7 * DAY);
                if frequency == Frequency::W {
                    assert_eq!(fields(rolled).weekday, 6);
                    assert_eq!((start - rolled) % DAY, 0);
                } else {
                    assert_eq!(rolled, start);
                }
            }
        }
    }
}
