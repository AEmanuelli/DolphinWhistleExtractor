from count_whistles import count_whistle_groups, howmanywhistlesinthisfile


def test_count_whistle_groups():
    assert count_whistle_groups([(0, 0.4)]) == 1
    assert count_whistle_groups([(0, 0.4)] * 3) == 1
    assert count_whistle_groups([(0, 0.4)] * 4) == 2
    assert count_whistle_groups([(0, 0.4)] * 7) == 3
    assert count_whistle_groups([(0, 0.4)] * 10) == 4


def test_howmanywhistlesinthisfile(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(
        "file_name,initial_point,finish_point,confidence\n"
        "r.wav,0.0,0.4,0.9\n"
        "r.wav,0.4,0.8,0.9\n"   # consecutive with the previous one
        "r.wav,10.0,10.4,0.9\n"  # new group
    )
    assert howmanywhistlesinthisfile(str(path)) == 2
    assert howmanywhistlesinthisfile(str(tmp_path / "missing.csv")) == -1
