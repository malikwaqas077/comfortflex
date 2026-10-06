import pytest

from comfortflex.comfort import (ComfortInputs, adaptive_band, pmv_band, pmv_ppd,
                                 running_mean_outdoor, temperature_for_pmv)

# ISO 7730:2005 Annex D, Table D.1 reference cases: ta, tr, vel, rh, met, clo -> PMV, PPD
ISO_7730_CASES = [
    (22.0, 22.0, 0.1, 60, 1.2, 0.5, -0.75, 17),
    (27.0, 27.0, 0.1, 60, 1.2, 0.5, 0.77, 17),
    (27.0, 27.0, 0.3, 60, 1.2, 0.5, 0.44, 9),
    (23.5, 25.5, 0.1, 60, 1.2, 0.5, -0.01, 5),
    (23.5, 25.5, 0.3, 60, 1.2, 0.5, -0.55, 11),
    (19.0, 19.0, 0.1, 40, 1.2, 1.0, -0.60, 13),
    (23.5, 23.5, 0.3, 40, 1.2, 1.0, 0.12, 5),
    (22.0, 22.0, 0.1, 60, 1.6, 0.5, 0.05, 5),
    (27.0, 27.0, 0.1, 60, 1.6, 0.5, 1.17, 34),
    (27.0, 27.0, 0.3, 60, 1.6, 0.5, 0.95, 24),
]


@pytest.mark.parametrize("ta,tr,vel,rh,met,clo,pmv_ref,ppd_ref", ISO_7730_CASES)
def test_pmv_matches_iso_7730_reference(ta, tr, vel, rh, met, clo, pmv_ref, ppd_ref):
    pmv, ppd = pmv_ppd(ta, tr, vel, rh, met, clo)
    assert pmv == pytest.approx(pmv_ref, abs=0.02)
    assert ppd == pytest.approx(ppd_ref, abs=1.0)


def test_ppd_minimum_is_five_percent_at_neutral():
    p = ComfortInputs()
    t0 = temperature_for_pmv(0.0, p)
    _, ppd = pmv_ppd(t0, t0, p.vel, p.rh, p.met, p.clo)
    assert ppd == pytest.approx(5.0, abs=0.01)


def test_pmv_band_brackets_neutral_and_widens_with_limit():
    p = ComfortInputs(clo=0.5)
    lo_b, hi_b = pmv_band(p, 0.5)
    lo_a, hi_a = pmv_band(p, 0.2)
    assert lo_b < lo_a < temperature_for_pmv(0.0, p) < hi_a < hi_b


def test_more_clothing_lowers_comfort_band():
    assert pmv_band(ComfortInputs(clo=1.0))[0] < pmv_band(ComfortInputs(clo=0.5))[0]


def test_adaptive_band_en16798():
    lo, hi = adaptive_band(20.0, "II")
    t_comf = 0.33 * 20 + 18.8
    assert (lo, hi) == pytest.approx((t_comf - 4, t_comf + 3))
    assert running_mean_outdoor([10.0, 10.0, 10.0]) == pytest.approx(10.0)
