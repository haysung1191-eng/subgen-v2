from subgen.calibration import TimingCalibrator
from subgen.types import SubtitleSegment


def test_calibration_repairs_overlap_conservatively() -> None:
    segments = [
        SubtitleSegment(start=0.0, end=1.0, text="a"),
        SubtitleSegment(start=0.8, end=1.4, text="b"),
    ]
    result = TimingCalibrator().cleanup(segments, min_duration_sec=0.2, hard_gap_sec=0.06, global_shift_ms=0)
    assert len(result.segments) == 2
    assert result.segments[0].end + 0.06 <= result.segments[1].start
    assert result.segments[1].start == 0.8
    assert result.overlap_repair_count == 1
    assert result.overlap_end_trim_count == 1
    assert result.avg_end_trim_ms > 0.0


def test_calibration_falls_back_to_start_shift_when_trim_is_not_enough() -> None:
    segments = [
        SubtitleSegment(start=0.0, end=0.21, text="a"),
        SubtitleSegment(start=0.1, end=0.3, text="b"),
    ]
    result = TimingCalibrator().cleanup(segments, min_duration_sec=0.2, hard_gap_sec=0.06, global_shift_ms=0)
    assert result.overlap_repair_count == 1
    assert result.segments[1].start == 0.1


def test_cleanup_does_not_move_later_start_for_end_policy_overlap() -> None:
    baseline = [
        SubtitleSegment(start=0.0, end=0.8, text="a"),
        SubtitleSegment(start=1.0, end=1.6, text="b"),
    ]
    end_policy_variant = [
        SubtitleSegment(start=0.0, end=1.05, text="a"),
        SubtitleSegment(start=1.0, end=1.6, text="b"),
    ]
    baseline_result = TimingCalibrator().cleanup(baseline, min_duration_sec=0.2, hard_gap_sec=0.06, global_shift_ms=0)
    variant_result = TimingCalibrator().cleanup(end_policy_variant, min_duration_sec=0.2, hard_gap_sec=0.06, global_shift_ms=0)
    assert baseline_result.segments[1].start == variant_result.segments[1].start
    assert variant_result.segments[1].start == 1.0
