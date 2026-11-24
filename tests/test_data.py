import pytest

from mfp.data import (AI4IDataset, DataValidationError, FAILURE_MODES, LEAKY_OR_ID, RAW_FEATURES,
                      DATA_PATH)

EXPECTED_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"


@pytest.fixture(scope="module")
def ds():
    return AI4IDataset()


def test_file_is_the_documented_copy(ds):
    assert ds.sha256() == EXPECTED_SHA256


def test_shape_and_label_rate(ds):
    rep = ds.report()
    assert rep.rows == 10_000
    assert rep.failures == 339
    assert rep.type_counts == {"L": 6000, "M": 2997, "H": 1003}
    assert rep.missing_values == 0 and rep.duplicate_rows == 0


def test_label_quirks_are_reported(ds):
    rep = ds.report()
    assert rep.failure_without_mode == 9
    assert rep.mode_without_failure == 18
    assert any("18 RNF" in n for n in rep.notes)


def test_features_exclude_ids_and_failure_modes(ds):
    X, y = ds.features_and_target()
    assert X.shape == (10_000, 6) and y.shape == (10_000,)
    assert not set(RAW_FEATURES) & set(LEAKY_OR_ID)
    assert not set(RAW_FEATURES) & set(FAILURE_MODES)
    assert set(y) == {0, 1}


def _write_csv(tmp_path, drop=None, bad_type=False):
    lines = DATA_PATH.read_text(encoding="utf-8-sig").splitlines()[:6]
    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]
    if bad_type:
        rows[0][2] = "X"
    if drop:
        i = header.index(drop)
        header.pop(i)
        for r in rows:
            r.pop(i)
    path = tmp_path / "data.csv"
    path.write_text("\n".join(",".join(r) for r in [header] + rows))
    return path


def test_rejects_missing_column(tmp_path):
    with pytest.raises(DataValidationError, match="missing columns"):
        AI4IDataset(_write_csv(tmp_path, drop="Torque [Nm]")).load()


def test_rejects_unknown_product_type(tmp_path):
    with pytest.raises(DataValidationError, match="unknown product types"):
        AI4IDataset(_write_csv(tmp_path, bad_type=True)).load()
