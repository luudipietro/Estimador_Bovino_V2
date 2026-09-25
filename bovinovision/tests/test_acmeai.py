from common import acmeai


def test_parsear_nombres_reales():
    assert acmeai.parsear_nombre("B3", "100_s_181_F.jpg") == {
        "animal_id": "100", "subbatch": None, "peso_kg": 181.0, "sexo": "F"}
    assert acmeai.parsear_nombre("B4", "100_b4-2_s_149_F.jpg") == {
        "animal_id": "100", "subbatch": "2", "peso_kg": 149.0, "sexo": "F"}


def test_nombre_irregular_devuelve_none():
    assert acmeai.parsear_nombre("B3", "IMG_20220129_104320.jpg") is None
    assert acmeai.parsear_nombre("B4", "100_s_181_F.jpg") is None


def test_kp_por_nombre_independiente_del_orden():
    # B3 y B4 listan los mismos puntos en distinto orden: el resultado por nombre debe coincidir
    b3 = ["1_wither", "5_front_girth_bottom", "4_front_girth_top"]
    b4 = ["1_wither", "4_front_girth_top", "5_front_girth_bottom"]
    a = acmeai.kp_por_nombre([1, 2, 2, 3, 4, 2, 5, 6, 2], b3)
    b = acmeai.kp_por_nombre([1, 2, 2, 5, 6, 2, 3, 4, 2], b4)
    assert a == b
    assert a["front_girth_bottom"] == (3.0, 4.0, 2)


def test_nombres_canonicos_cubren_ambos_lotes():
    b3 = ['1_wither', '2_pinbone', '3_shoulderbone', '5_front_girth_bottom', '4_front_girth_top',
          '9_Height_bottom', '8_Height_top', '7_rear_girth_bottom', '6_rear_girth_top']
    b4 = ['1_wither', '2_pinbone', '3_shoulderbone', '4_front_girth_top', '5_front_girth_bottom',
          '6_rear_girth_top', '7_rear_girth_bottom', '8_Height_top', '9_Height_bottom']
    for nombres in (b3, b4):
        assert sorted(acmeai.nombre_kp_canonico(n) for n in nombres) == sorted(acmeai.KP_CANONICOS)
